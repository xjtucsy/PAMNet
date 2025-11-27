"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
import math
from timm.models.layers import trunc_normal_

from archs.mamba_layer import Mamba, MambaConfig


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)
class DPSConv3d(nn.Module):
    def __init__(self, in_channels):
        # out_channels = in_channels
        # DW-PW
        super(DPSConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, [1, 1, 1], stride=1, padding=[0, 0, 0], bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, [3, 1, 1], stride=1, padding=[1, 0, 0], groups=in_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, [1, 1, 1], stride=1, padding=[0, 0, 0], bias=False),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class INN(nn.Module):
    # out_channels = 2 * in_channels
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.expand_layer = nn.Conv3d(in_channels, in_channels*2, [3, 1, 1], stride=1, padding=[1, 0, 0], groups=in_channels)
        self.I1 = DPSConv3d(in_channels)
        self.I2 = DPSConv3d(in_channels)
        self.I3 = DPSConv3d(in_channels)
        
    def forward(self, x):
        x_expand = self.expand_layer(x) # [B, 2C, T, H, W]
        x1, x2 = torch.chunk(x_expand, 2, dim=1)
        I1_x1 = self.I1(x1)
        x2 = x2 + I1_x1
        I2_x2 = self.I2(x2)
        I3_x2 = self.I3(x2)
        x1 = x1*torch.exp(I2_x2) + I3_x2
        return torch.cat([x1, x2], dim=1)

class STEM(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(STEM, self).__init__()
        # Conv + DW-PW + DW-PW
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//4, [1, 3, 3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            
            INN(out_channels//4),
            # nn.Conv3d(out_channels//4, out_channels//2, [3, 3, 3], stride=1, padding=1, groups=out_channels//4),
            # nn.Conv3d(out_channels//2, out_channels//2, [1, 1, 1], [1, 1, 1], [0, 0, 0]),
            nn.BatchNorm3d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            INN(out_channels//2),
            # nn.Conv3d(out_channels//2, out_channels, [3, 3, 3], stride=1, padding=1, groups=out_channels//2),
            # nn.Conv3d(out_channels, out_channels, [1, 1, 1], [1, 1, 1], [0, 0, 0]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        
        self.apply(self.init_weights)
        
    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        x : [B, C, T, H, W] = [B, 3, 160, 128, 128]
        stem_feat : [B, dim, 160, 32, 32]
        '''
        #print(f'x_before_stem: {x.shape}')
        stem_feat = self.conv(x)
        #print(f'stem_feat: {stem_feat.shape}')
        return stem_feat
    
    
class SignalExtractor(nn.Module):
    '''
        从特征图中提取信号,特征图 [B, C, T, H, W] = [B, 64, 40, 4, 4]
        1. 3D Upsample -> [B, 64, 160, 4, 4]
        2. frame squeeze * 2 -> [B, 64, 160]
        3. channel squeeze -> [B, 1, 160] -> [B, 160]
    '''
    def __init__(self, out_channels, T):
        super(SignalExtractor, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(size=(T//2, 4, 4)),
            nn.Conv3d(out_channels, out_channels, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(out_channels),
            nn.ELU(),
            
            nn.Upsample(size=(T, 4, 4)),
            nn.Conv3d(out_channels, out_channels//2, [3, 1, 1], stride=1, padding=(1,0,0)),   
            nn.BatchNorm3d(out_channels//2),
            nn.ELU(),
        )
        
        self.channel_squeeze = nn.Conv1d(out_channels//2, 1, 1,stride=1, padding=0)
        
    def forward(self, x):
        x_upsample = self.upsample(x)
        x_frame_squeeze = torch.mean(x_upsample, 4)
        x_frame_squeeze = torch.mean(x_frame_squeeze, 3)    # [B, 64, 160]
        x_channel_squeeze = self.channel_squeeze(x_frame_squeeze)  # [B, 1, 160]
        rPPG = x_channel_squeeze.squeeze(1)  # [B, 160]
        return rPPG


# stem_3DCNN + ST-ViT with local Depthwise Spatio-Temporal MLP
class PAMNet(nn.Module):

    def __init__(
        self, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_layers: int = 12,
        dropout_rate: float = 0.2,
        in_channels: int = 3, 
        frame: int = 160,
        image_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        
        self.frame = frame
        self.dim = dim
        
        # Image and patch sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40
        
        # Stem Block
        self.stem = STEM(in_channels, dim)

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))
        
        #! Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 16, dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, 40, dim))
        self.pos_dropout = nn.Dropout(p=dropout_rate)
        
        # mamba
        mamba_config = MambaConfig(d_model=dim, n_layers=num_layers,
                                   ff_dim=ff_dim, dropout=dropout_rate,
                                   device=device)
        self.Mamba_block = Mamba(mamba_config)
        
        self.signal_extractor = SignalExtractor(dim, image_size[0])
    
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        trunc_normal_(self.pos_embedding, std=.02)


    def forward(self, inputs):
        x = inputs['input_clip']
        epoch = inputs['epoch']
        
        x = torch.diff(x, n=1, dim=2, prepend=x[:, :, 0:1, :, :])  # [B, C, T, H, W]

        b, c, t, fh, fw = x.shape
        
        stem_feat = self.stem(x)  # [B, dim, 160, 32, 32]
        
        #print(f'x_after_stem: {stem_feat.shape}')
        
        x = self.patch_embedding(stem_feat)  # [B, dim, 40, 4, 4]
        #print(f"x_after_patch_embedding1: {x.shape}")
        
        #! Positional embedding
        x = rearrange(x, 'b c t fh fw -> (b t) (fh fw) c')
        x = x + self.pos_embedding / 255
        x = rearrange(x, '(b t) n c -> (b n) t c', t=t//4)
        if t // 4 != 40:
            # ! upsample temporal dimension for temporal positional embedding
            cur_temporal_pos_embedding = F.interpolate(self.temporal_pos_embedding.permute(0, 2, 1), size=(t//4), mode='linear')
            x = x + cur_temporal_pos_embedding.permute(0, 2, 1) / 255
        else:
            x = x + self.temporal_pos_embedding / 255
        x = rearrange(x, '(b n) t c -> b (t n) c', n=16)
        x = self.pos_dropout(x)        
        # print(f"x_after_patch_embedding: {x.shape}")
        
        # trans_features = x
        trans_features, offset_map = self.Mamba_block(x, epoch)     
        #print(f"x_after_patch_embedding: {x.shape}")
                
        #print(f'trans_features.shape : {trans_features.shape}')
        
        # upsampling heads
        features_last = trans_features.transpose(1, 2).view(b, self.dim, t//4, 4, 4) # [B, 64, 40, 4, 4]
        
        rPPG = self.signal_extractor(features_last)  # [B, 160]
        
        #print(f'rPPG.shape : {rPPG.shape}')
        outputs = {
            'rPPG': rPPG,
            'offset_map' : offset_map
        }
        return outputs


if __name__ == '__main__':
    #! cal params. and MACs
    from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

    model = PAMNet(image_size=(160, 128, 128), patches=(4,4,4),\
            dim=32, ff_dim=64, num_layers=6, dropout_rate=0.1).cuda()
    model.eval()
    input_data = {'input_clip' : torch.randn(1, 3, 160, 128, 128).cuda(), 'epoch' : 0}
    # import time
    # start_time = time.time()
    # for i in range(100):
    #     output = model(input_data)['rPPG']
    # time.sleep(5)
    # end_time = time.time()
    # print(f'cost time: {(end_time-start_time)/160/100/4 * 1000} ms')
    output = model(input_data)['rPPG']
    print(f'output.shape: {output.shape}')
    prof = FlopsProfiler(model)
    prof.start_profile(ignore_list=[type(nn.Upsample())])
    output = model(input_data)['rPPG']
    params = prof.get_total_params(as_string=True)
    flops = prof.get_total_macs(as_string=True)
    print(f'MACs: {flops}, Params: {params}')
    print(f'output.shape: {output.shape}')
    prof.end_profile()

    