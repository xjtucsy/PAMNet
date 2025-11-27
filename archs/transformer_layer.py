"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch
import math

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
                
        self.proj_q = nn.Sequential(
            # CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.Conv3d(dim, dim, 3, stride=[2,1,1], padding=1, groups=dim, bias=False),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            # CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),  
            nn.Conv3d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        
        # self.down_sample = nn.Sequential(
        #     nn.Conv3d(dim, dim, 3, stride=[2,1,1], padding=1, groups=dim, bias=False),
        #     nn.BatchNorm3d(dim),
        # )
        
        self.up_sample = nn.Sequential(
            nn.ConvTranspose3d(dim, dim, 3, stride=[2,1,1], padding=1, output_padding=[1,0,0], groups=dim, bias=False),
            nn.BatchNorm3d(dim),
        )
    

    def forward(self, x):    # [B, 40*4*4, 128]
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        [B, P, C]=x.shape
        x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
        # print(f'x_down: {x_down.shape}')
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)  # [B, 20*2*2, dim//2]
        k = k.flatten(2).transpose(1, 2)  # [B, 40*4*4, dim//2]
        v = v.flatten(2).transpose(1, 2)  # [B, 40*4*4, dim//2]
        
        # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h = self.n_heads) for x in [q, k, v]]
        # (B, H, N/8, D) @ (B, H, D, N) -> (B, H, N/8, N) -softmax-> (B, H, N/8, N)
        # print(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')
        attn_map = q @ k.transpose(-2, -1) # [B, H, N/8, N]
        # print(f'attn: {attn_map.shape}')

        attn = self.drop(F.softmax(attn_map, dim=-1))   # [B, H, S/8, S]
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        out_down = attn @ v  # [B, H, S/8, W]
        out_down = rearrange(out_down, 'b hd (t h w) d -> b (hd d) t h w', h=4, w=4)
        out = self.up_sample(out_down) # [B, dim, 40, 4, 4]
        # -merge-> (B, S, D)
        # print(f'out: {out.shape}')
        out = rearrange(out, 'b d t h w -> b (t h w) d')
        # print(f'out: {out.shape}')
        return out, attn_map.sum(1).sum(1).reshape(B, P)


# class PositionWiseFeedForward_ST(nn.Module): #! normal FFN
#     def __init__(self, dim, ff_dim, K, S):
#         super().__init__()
#         self.fc1 = nn.Linear(dim, ff_dim)  # 第一层：输入到隐藏层
#         self.relu = nn.ReLU()  # 激活函数：ReLU
#         self.fc2 = nn.Linear(ff_dim, dim)  # 第二层：隐藏层到输出层
#         self.sigmoid = nn.Sigmoid()  # 输出层的激活函数：Sigmoid用于二分类

#     def forward(self, x, epoch):
#         x = self.fc1(x)  # 输入经过第一层
#         x = self.relu(x)  # 激活函数
#         x = self.fc2(x)  # 经过第二层
#         x = self.sigmoid(x)  # 输出通过Sigmoid
#         return x


# class PositionWiseFeedForward_ST(nn.Module):
#     """FeedForward Neural Networks for each position"""
#     def __init__(self, dim, ff_dim, K, S):
#         super().__init__()
        
#         self.fc1 = nn.Sequential(
#             nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),  
#             nn.BatchNorm3d(ff_dim),
#             nn.ELU(),
#         )
        
#         self.STConv = nn.Sequential(
#             nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),  
#             nn.BatchNorm3d(ff_dim),
#             nn.ELU(),
#         )
        
#         self.fc2 = nn.Sequential(
#             nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),  
#             nn.BatchNorm3d(dim),
#         )

#     def forward(self, x, epoch):    # [B, 4*4*40, 128]
#         [B, P, C]=x.shape
#         #x = x.transpose(1, 2).view(B, C, 40, 4, 4)      # [B, dim, 40, 4, 4]
#         x = x.transpose(1, 2).view(B, C, P//16, 4, 4)      # [B, dim, 40, 4, 4]
#         x = self.fc1(x)		              # x [B, ff_dim, 40, 4, 4]
#         x = self.STConv(x)		          # x [B, ff_dim, 40, 4, 4]
#         x = self.fc2(x)		              # x [B, dim, 40, 4, 4]
#         x = x.flatten(2).transpose(1, 2)  # [B, 4*4*40, dim]
        
#         return x
    
class PositionWiseFeedForward_ST(nn.Module):
    def __init__(self, dim_in, dim_out, K, S):
        super(PositionWiseFeedForward_ST, self).__init__()
        # dim_out = dim_out * K // S
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.K = K
        self.S = S
        self.weight = nn.Parameter(torch.empty(2, dim_out, dim_in), requires_grad=True)
        # self.middle_dim = dim_out
        self.conv_weight = nn.Parameter(torch.empty(dim_out, 1, 3, 3, 3), requires_grad=True)
        # self.bn_conv = nn.BatchNorm3d(dim_out)
        self.bn_final = nn.BatchNorm3d(dim_in)
        self.init_weight()
        # print(f'weight : {self.weight.shape}')
        
    def init_weight(self): 
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.trunc_normal_(self.conv_weight, std=0.02)

    def hash_func(self, x : torch.Tensor, K : int) -> torch.Tensor:
        # print(f'x : {x.shape}')
        x_mean = x.mean(dim=0, keepdim=True)
        hash_value = torch.matmul(x, x_mean.t()) / math.sqrt(self.dim_in)
        hash_value = hash_value.squeeze()
        hash_value = hash_value % K
        return hash_value
    
    def forward(self, x : torch.Tensor, epoch : int):
        B, N, C = x.shape
        mlp_weight1 = self.weight[0]
        mlp_weight2 = self.weight[1].transpose(0, 1)
        conv_weight = self.conv_weight
        inner_dim = self.dim_out
        epoch = 0
        if epoch >= 7: 
            # # sort weight by hash_value
            # print(f'x : {x.shape}, weight : {self.weight.shape}')
            hash_value = self.hash_func(self.weight[0], self.K) # (D)
            _, indices = torch.sort(hash_value) # (D)
            weight = self.weight[:, indices, :] # (2, D, C)
            # divide weight into K buckets
            weight_buckets = weight.view(2, self.K, -1, self.dim_in) # (2, K, D/K, C)
            conv_weight_buckets = self.conv_weight.view(self.K, -1, 1, 3, 3, 3) # (K, D/K, 1, 3, 3, 3)
            # print(f'weight_buckets : {weight_buckets.shape}') 
            # cal proxy feat for each bucket
            proxy_bucket = torch.mean(weight_buckets[0], dim=1).squeeze() # (K, C)
            # cal dist for x and proxy_bucket
            dist = torch.matmul(x, proxy_bucket.t()).sum(0).sum(0)    # (K)
            # select top-k dist
            selected_bucket = torch.topk(dist, self.S, dim=-1).indices # (S)
            # print(f'selected_bucket : {selected_bucket.shape}')
            # get selected weight from weight_buckets
            weight_buckets = weight_buckets.transpose(0, 1)
            selected_weight = weight_buckets[selected_bucket]   # (S, 2, D/K, C)
            # # print(f'selected_weight : {selected_weight.shape}')
            # combine selected_weight
            selected_weight = rearrange(selected_weight, 's m d c -> m (s d) c')
            
            # # MLP for x and selected_weight
            mlp_weight1 = selected_weight[0] # (S*D, C)
            conv_weight = conv_weight_buckets[selected_bucket] # (S, D/K, 1, 3, 3, 3)
            conv_weight = conv_weight.view(-1, 1, 3, 3, 3) # (S*D, 1, 3, 3, 3)
            mlp_weight2 = selected_weight[1].transpose(0, 1) # (C, S*D)
            
            inner_dim = self.dim_out * self.S // self.K

        # x = x.unsqueeze(-1).transpose(2, 3) # (B, N, C, 1)
        # hidden_feat = torch.einsum('b n c, d c -> b n d', x, mlp_weight1) # (B, N, S*D)
        x = rearrange(x, 'b (t h w) d -> b d t h w', t=N//16, h=4, w=4)
        hidden_feat = F.conv3d(x, mlp_weight1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), stride=1, padding=0, groups=1)
        hidden_feat = F.relu(hidden_feat)
        # hidden_feat = rearrange(hidden_feat, 'b (t h w) d -> b d t h w', t=N//16, h=4, w=4)
        hidden_feat = F.conv3d(hidden_feat, conv_weight, stride=1, padding=1, groups=inner_dim)
        # hidden_feat = self.bn_conv(hidden_feat)
        hidden_feat = F.conv3d(hidden_feat, mlp_weight2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), stride=1, padding=0, groups=1)
        hidden_feat = self.bn_final(hidden_feat)
        hidden_feat = rearrange(hidden_feat, 'b d t h w -> b (t h w) d') # (B, N, S*D)
        # hidden_feat = torch.einsum('b n d, d c -> b n c', hidden_feat, mlp_weight2)
        return hidden_feat