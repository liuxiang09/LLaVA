import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2

# Vision Tower 构建器实现解析：
# 这段代码实现了一个视觉编码器（Vision Tower）的构建工厂函数，主要用于 LLaVA 多模态模型中处理视觉输入的部分。

def build_vision_tower(vision_tower_cfg, **kwargs):
    # 获取视觉塔的配置，优先使用 'mm_vision_tower'，其次是 'vision_tower'
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    # 检查视觉塔路径是否存在
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    # 检查是否使用 S2 版本
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    # 如果路径存在或视觉塔名称以 "openai" 或 "laion" 开头，或包含 "ShareGPT4V"
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            # 使用 CLIPVisionTowerS2 构建视觉塔
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            # 使用 CLIPVisionTower 构建视觉塔
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # 如果视觉塔名称未知，抛出异常
    raise ValueError(f'Unknown vision tower: {vision_tower}')
