import torch
import torch.nn as nn
import re


# 一个简单的恒等映射层
class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


# 一个简单的残差块
class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)  # 层归一化

        # 一个简单的 MLP
        self.proj = nn.Sequential(
            nn.Linear(channels, channels), # 线性变换
            nn.GELU(), # GELU 激活函数
            nn.Linear(channels, channels) # 线性变换
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x) # 残差连接


# 构建视觉投影器
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear') # 获取视觉投影器类型，默认为线性投影器

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1)) # 获取 MLP 深度 
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)] # 线性变换
        for _ in range(1, mlp_depth): # 构建 MLP
            modules.append(nn.GELU()) # GELU 激活函数
            modules.append(nn.Linear(config.hidden_size, config.hidden_size)) # 线性变换
        return nn.Sequential(*modules) # 返回序列模块
    
    # 如果视觉投影器类型为 'identity'，则返回一个恒等映射层
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
