import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModel, LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ------------------
# 1. 模型架构定义
# ------------------
class MultitaskCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # 原始CLIP视觉编码器 + LoRA
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["attn.k_proj", "attn.v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_config)
        
        # 新增任务头
        self.detect_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4+80, 1)  # 4坐标 + 80类
        )
        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(768, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 80, 1)  # 80类掩码
        )

    def forward(self, pixel_values):
        # 视觉特征提取
        outputs = self.clip(pixel_values, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [batch, 50, 768]
        
        # 转换为2D特征图 (假设输入为224x224，patch_size=32)
        batch_size = last_hidden.size(0)
        features_2d = last_hidden[:, 1:].view(batch_size, 7, 7, 768).permute(0,3,1,2)
        
        # 任务输出
        det_output = self.detect_head(features_2d)
        seg_output = self.seg_head(features_2d)
        
        return {
            "det_bbox": det_output[:, :4],  # [B,4,7,7]
            "det_cls": det_output[:, 4:],    # [B,80,7,7]
            "seg_mask": seg_output          # [B,80,14,14]
        }

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = MultitaskCLIP()
        self.llm = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        self.tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        
        # 冻结LLM参数
        for param in self.llm.parameters():
            param.requires_grad = False
            
    def forward(self, images, texts):
        # 视觉编码
        visual_outputs = self.visual_encoder(images)
        
        # 文本编码
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(images.device)
        
        # 简单拼接（实际应使用投影层）
        llm_outputs = self.llm(
            input_ids=input_ids,
            inputs_embeds=visual_outputs["det_cls"].mean(dim=[2,3]).unsqueeze(1)
        )
        
        return {
            "logits": llm_outputs.logits,
            "det_bbox": visual_outputs["det_bbox"],
            "det_cls": visual_outputs["det_cls"],
            "seg_mask": visual_outputs["seg_mask"]
        }

# ------------------
# 2. 合成数据集
# ------------------
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 随机生成假数据
        return {
            "image": torch.randn(3, 224, 224),  # 假图像
            "text": "Describe the image",       # 假文本
            "det_labels": torch.randint(0,80,(5,)), # 假检测标签
            "seg_mask": torch.randint(0,80,(14,14)) # 假分割掩码
        }

# ------------------
# 3. 训练循环
# ------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化
    model = MultimodalModel().to(device)
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2)
    
    # 只训练视觉部分的LoRA参数和任务头
    optimizer = torch.optim.Adam([
        {'params': model.visual_encoder.parameters()},
    ], lr=1e-4)
    
    # 简单损失函数
    det_criterion = nn.MSELoss()
    seg_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(3):  # 3个epoch演示
        for batch in loader:
            images = batch["image"].to(device)
            
            # 前向传播
            outputs = model(images, texts=["describe"]*len(images))
            
            # 计算检测损失（伪标签）
            det_loss = det_criterion(outputs["det_bbox"], torch.randn_like(outputs["det_bbox"]))
            
            # 计算分割损失（伪标签）
            seg_loss = seg_criterion(
                outputs["seg_mask"], 
                torch.randint(0,80, outputs["seg_mask"].shape[1:]).to(device)
            )
            
            # 总损失
            loss = det_loss + seg_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
    print("训练完成！可以尝试推理：")
    
    # 简单推理演示
    model = MultimodalModel().eval()
    dummy_image = torch.randn(1,3,224,224)
    with torch.no_grad():
        outputs = model(dummy_image, texts=["What objects are present?"])
        print("检测输出形状：", outputs["det_bbox"].shape)
        print("分割输出形状：", outputs["seg_mask"].shape)