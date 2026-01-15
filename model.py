# 模型定义文件：包含牛脸识别系统的模型架构
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 主干网络定义
def get_backbone(name='resnet50', pretrained=True):
    """
    获取主干网络（特征提取器）
    
    参数:
        name: 主干网络名称，目前仅支持'resnet50'
        pretrained: 是否使用预训练权重
    
    返回:
        backbone: 主干网络模型
        embedding_size: 主干网络输出特征维度
    """
    if name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
    else:
        raise NotImplementedError(f"Backbone {name} not supported")
    
    # 移除原始分类层
    embedding_size = backbone.fc.in_features
    backbone.fc = nn.Identity()  # 直接输出 global average pooling 后的特征
    
    return backbone, embedding_size

# 嵌入模型定义
class EmbeddingModel(nn.Module):
    """
    牛脸特征提取模型：Backbone + Neck
    输出 L2 归一化的 embedding
    """
    def __init__(self, backbone_name='resnet50', embedding_size=512):
        super(EmbeddingModel, self).__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained=True)
        
        # Neck: 特征映射层
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        
        # 初始化 neck
        for m in self.neck.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        输入图像，输出归一化后的 embedding
        
        参数:
            x: 输入图像张量，形状为 [B, 3, H, W]
        
        返回:
            归一化后的特征嵌入，形状为 [B, embedding_size]
        """
        feat = self.backbone(x)  # (B, C)
        embed = self.neck(feat)  # (B, embedding_size)
        return nn.functional.normalize(embed, p=2, dim=1)

# 获取模型
def get_model(config):
    """
    根据配置创建并返回嵌入模型
    
    参数:
        config: 配置字典，包含模型相关参数
    
    返回:
        初始化好的嵌入模型，已移至可用设备
    """
    model = EmbeddingModel(
        backbone_name=config['model']['backbone'],
        embedding_size=config['model']['embedding_size']
    )
    return model.cuda() if torch.cuda.is_available() else model