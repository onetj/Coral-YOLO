import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PolarLinearAttention(nn.Module):
    """
    Polar Linear Attention模块，适用于YOLOv12
    结合了极坐标变换和线性注意力机制，提高特征表示能力
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 查询、键、值的线性变换
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 极坐标变换参数
        self.polar_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.polar_norm = nn.LayerNorm(dim)

        # 线性注意力的特征映射
        self.feature_map = nn.ReLU()

    def polar_transform(self, x):
        """
        极坐标变换：将笛卡尔坐标系下的特征转换为极坐标表示
        """
        B, C, H, W = x.shape

        # 创建坐标网格
        y_coords = torch.arange(H, dtype=torch.float32, device=x.device).view(-1, 1).repeat(1, W)
        x_coords = torch.arange(W, dtype=torch.float32, device=x.device).view(1, -1).repeat(H, 1)

        # 中心化坐标
        center_y, center_x = H // 2, W // 2
        y_coords = y_coords - center_y
        x_coords = x_coords - center_x

        # 计算极坐标
        rho = torch.sqrt(x_coords ** 2 + y_coords ** 2)
        theta = torch.atan2(y_coords, x_coords)

        # 归一化
        rho = rho / (max(H, W) / 2)
        theta = (theta + math.pi) / (2 * math.pi)  # 归一化到[0,1]

        # 极坐标特征增强
        polar_feat = torch.stack([rho, theta], dim=0).unsqueeze(0).repeat(B, C // 2, 1, 1)

        # 与原始特征结合
        if C % 2 != 0:
            # 如果通道数为奇数，复制最后一个通道
            extra_feat = polar_feat[:, -1:, :, :]
            polar_feat = torch.cat([polar_feat, extra_feat], dim=1)

        enhanced_x = x + polar_feat
        enhanced_x = self.polar_conv(enhanced_x)

        return enhanced_x

    def linear_attention(self, q, k, v):
        """
        线性注意力计算，复杂度为O(n)而非O(n²)
        """
        B, H, N, D = q.shape

        # 特征映射
        q = self.feature_map(q)
        k = self.feature_map(k)

        # 计算线性注意力
        k_cumsum = k.sum(dim=-2, keepdim=True)
        D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True).clamp(min=1e-6)

        context = k.transpose(-2, -1) @ v
        out = (q @ context) * D_inv

        return out

    def forward(self, x):
        B, C, H, W = x.shape

        # 极坐标变换
        x_polar = self.polar_transform(x)

        # 展平为序列格式
        x_flat = x_polar.flatten(2).transpose(1, 2)  # B, N, C
        x_flat = self.polar_norm(x_flat)

        N = x_flat.shape[1]

        # 生成Q, K, V
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, H, N, D

        # 应用线性注意力
        attn_output = self.linear_attention(q, k, v)

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        # 输出投影
        output = self.proj(attn_output)
        output = self.proj_drop(output)

        # 恢复到原始形状
        output = output.transpose(1, 2).reshape(B, C, H, W)

        # 残差连接
        return output + x


class PolarAttentionBlock(nn.Module):
    """
    完整的Polar Attention Block，包含前馈网络和残差连接
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = PolarLinearAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=attn_drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 注意力分支
        x_flat = x.flatten(2).transpose(1, 2)  # B, N, C
        x_norm1 = self.norm1(x_flat).transpose(1, 2).reshape(B, C, H, W)
        attn_out = self.attn(x_norm1)
        x = x + self.drop_path(attn_out)

        # MLP分支
        x_flat = x.flatten(2).transpose(1, 2)  # B, N, C
        x_norm2 = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm2).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.drop_path(mlp_out)

        return x


# 用于YOLOv12的轻量级版本
class PolarAttentionYOLO(nn.Module):
    """
    专为YOLO设计的轻量级Polar Attention模块
    """

    def __init__(self, c1, c2, num_heads=4, shortcut=True):
        super().__init__()
        self.c = c2
        self.num_heads = num_heads
        self.shortcut = shortcut and c1 == c2

        self.cv1 = nn.Conv2d(c1, c2, 1)
        self.cv2 = nn.Conv2d(c2, c2, 1)

        self.polar_attn = PolarLinearAttention(c2, num_heads=num_heads)

    def forward(self, x):
        if self.shortcut:
            return x + self.cv2(self.polar_attn(self.cv1(x)))
        else:
            return self.cv2(self.polar_attn(self.cv1(x)))


# 测试代码
if __name__ == "__main__":
    # 测试模块
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试PolarLinearAttention
    model = PolarLinearAttention(dim=256, num_heads=8).to(device)
    x = torch.randn(2, 256, 32, 32).to(device)

    print("输入形状:", x.shape)
    output = model(x)
    print("输出形状:", output.shape)

    # 测试YOLO版本
    yolo_model = PolarAttentionYOLO(256, 256, num_heads=4).to(device)
    yolo_output = yolo_model(x)
    print("YOLO版本输出形状:", yolo_output.shape)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")