import math
from dataclasses import dataclass
from typing import Union

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.pscan import pscan
from archs.transformer_layer import PositionWiseFeedForward_ST

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x), indexing = 'ij')
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output
    
@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 3
    dropout: float = 0.1
    topk: int = 8

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    
    window_num: int = 4

    rms_norm_eps: float = 1e-5
    frame_num: int = 160
    ff_dim: int = 128
    
    # pscan: bool = True # use parallel scan mode or sequential mode when training
    
    device: str = "cpu"

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, epoch):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x, offset_map = layer(x, epoch)

        return x, offset_map

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.drop = nn.Dropout(config.dropout)
        self.mixer = MambaBlock(config)
        # self.norm_mamba = RMSNorm(config.d_model, config.rms_norm_eps, config.device)
        self.norm_mamba = nn.LayerNorm(config.d_model, eps=1e-6)
        self.ffn = PositionWiseFeedForward_ST(config.d_model, config.ff_dim, 4, 2)
        # self.norm_ffn = RMSNorm(config.d_model, config.rms_norm_eps, config.device)
        self.norm_ffn = nn.LayerNorm(config.d_model, eps=1e-6)

    def forward(self, x, epoch):
        # x : (B, L, D)

        # output : (B, L, D)
        mamba_output, offset_map =  self.mixer(self.norm_mamba(x))
        mamba_output = mamba_output + x
        output = self.drop(self.ffn(self.norm_ffn(mamba_output), epoch)) + mamba_output
        
        return output, offset_map

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2*config.d_inner, bias=False)
        # input shape (B, D, T, H, W) -> conv3D -> (B, 2*ED, T, H, W) -> (B, L, 2*ED)
        # self.in_proj = nn.Sequential(
        #     nn.Conv3d(config.d_model, 2 * config.d_inner,
        #               kernel_size=3, stride=1, padding=1, groups=1, bias=False),
        #     nn.BatchNorm3d(2 * config.d_inner),
        #     nn.Sigmoid(),
        #     Rearrange('b ed t h w -> b (t h w) ed'),
        # )

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=True, 
                              groups=config.d_inner,
                              padding=(config.d_conv - 1) // 2)
        
        # self.freq_conv1d = nn.Conv1d(in_channels=config.d_model, out_channels=config.d_inner, 
        #                       kernel_size=config.d_conv, bias=True, 
        #                       groups=config.d_model,
        #                       padding=(config.d_conv - 1) // 2)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)).to(config.device) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner)).to(config.device)
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

        # fuse freq & st
        # self.conv_fusion = nn.Conv1d(in_channels=config.d_inner*2, out_channels=config.d_inner, 
        #                       kernel_size=1, bias=True, 
        #                       groups=config.d_inner,
        #                       padding=0)
        
        self.dt_layernorm = None
        self.B_layernorm = None
        self.C_layernorm = None
        
        #! spatial-offset generator
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 4, 4))
        self.conv1 = nn.Conv2d(2*config.d_model, config.d_model, kernel_size=3, stride=1, padding=1, bias=False, groups=config.d_model)
        self.norm = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.offset_conv = nn.Conv2d(config.d_model, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        B, L, _ = x.shape
        x_init = rearrange(x, 'b (t h w) d -> b d t h w', t=L//16, h=4, w=4) # (B, D, T, H, W)
        
        #!!! cal offset map from offset generator
        x_avg = self.avg_pool(x_init)   # (B, D, 1, 4, 4)
        x_max = self.max_pool(x_init)   # (B, D, 1, 4, 4)
        x_offset = torch.cat([x_avg, x_max], dim=1) # (B, 2D, 1, 4, 4)
        x_offset = self.conv1(x_offset.squeeze(2))  # (B, D, 4, 4)
        x_offset = self.norm(x_offset)
        offset_map = self.offset_conv(x_offset) # (B, 2, 4, 4)
        
        #! re-sample in spatial dimension
        x = rearrange(x, 'b (t h w) d -> b (t d) h w', t=L//16, h=4, w=4) # (BD, T, H, W)
        x = x + flow_warp(x, offset_map.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border')
        x_resample = rearrange(x, 'b (t d) h w -> b (t h w) d', t=L//16) # (B, L, D)
        
        #!!! spatial-mamba branch
        
        xz = self.in_proj(x_resample) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        #!! x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x) # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)
        
        #!! shuffle mamba
        x = F.silu(x) # (B, L, ED)
        
        #! shuffle
        x = rearrange(x, 'b (t n) d -> b t n d', n=16)
        if self.training:
            T = L // 16
            N = 16
            # spatial-random-shuffle
            spatial_idx = torch.randperm(N)
            shuffle_x = x[:, :, spatial_idx, :]
            # temporal-random-shift
            temporal_shift = torch.randint(0, T//2, (B, 1, N)) - T//4
            # roll the tensor w.r.t. temporal dimension
            for b in range(B):
                for n in range(N):
                    shuffle_x[b, :, n, :] = torch.roll(shuffle_x[b, :, n, :], 
                                                        shifts=temporal_shift[b, 0, n].item(), 
                                                        dims=0)
        else:
            shuffle_x = x
                
        #! bi-mamba
        x = rearrange(shuffle_x, 'b t n d -> b (t n) d')
        y_shuffle = self.ssm(x)
        x_bi = x.flip(1)
        y_shuffle_bi = self.ssm(x_bi)
        y_shuffle = y_shuffle + y_shuffle_bi.flip(1)
        y_shuffle = rearrange(y_shuffle, ' b (t n) d -> b t n d', n=16) # (B, T, N, ED)
        
        #! reverse shuffle
        if self.training:
            # temporal-shift reverse
            for b in range(B):
                for n in range(N):
                    y_shuffle[b, :, n] = torch.roll(y_shuffle[b, :, n, :], 
                                                            shifts=-temporal_shift[b, 0, n].item(), 
                                                            dims=0)
            # spatial-shuffle reverse
            reverse_spatial_idx = torch.argsort(spatial_idx)
            y_shuffle = y_shuffle[:, :, reverse_spatial_idx, :]
        else:
            y_shuffle = y_shuffle
            
        y = rearrange(y_shuffle, 'b t n d -> b (t n) d')
        
        #!! z branch attn_map B,P -> B,T,H*W -> gather -> B,T,topk -> B,L',1
        z = F.silu(z)
       
        # print(z.shape, y.shape)
        
        mamba_output = y * z  # [B, L, ED]
        mamba_output = self.out_proj(mamba_output) # (B, L, D)
        
        return mamba_output, offset_map
    
    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)
        
        _, L, _ = x.shape

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        # x_conv = rearrange(x, 'b (t h w) d -> b d t h w', t=L//16, h=4, w=4) # (B, ED, T, H, W)
        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        delta = delta.transpose(1, 2)
        delta = F.softplus(delta + self.dt_proj.bias)

        y = self.selective_scan(x, delta, A, B, C, D)
      
        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
   

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: str = 'cpu'):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model)).to(device=device)

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
