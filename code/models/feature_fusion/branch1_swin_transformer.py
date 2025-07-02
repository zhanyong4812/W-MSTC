# branch1_swin_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowPartition(nn.Module):
    """将序列划分为不重叠的窗口"""
    def __init__(self, window_size):
        super(WindowPartition, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        """
        x: [B, L, C]
        returns: [num_windows*B, window_size, C]
        """
        B, L, C = x.shape
        window_size = self.window_size
        num_windows = L // window_size
        if L % window_size != 0:
            # 如果序列长度不是窗口大小的整数倍，则进行填充
            pad_len = window_size - (L % window_size)
            x = F.pad(x, (0, 0, 0, pad_len), "constant", 0)
            B, L, C = x.shape
            num_windows = L // window_size
        windows = x.view(B, num_windows, window_size, C)  # [B, num_windows, window_size, C]
        windows = windows.view(-1, window_size, C)  # [num_windows*B, window_size, C]
        return windows

class WindowReverse(nn.Module):
    """将窗口恢复为序列"""
    def __init__(self, window_size, original_length):
        super(WindowReverse, self).__init__()
        self.window_size = window_size
        self.original_length = original_length

    def forward(self, x):
        """
        x: [num_windows*B, window_size, C]
        returns: [B, L, C]
        """
        B = int(x.shape[0] / (self.original_length // self.window_size))
        window_size = self.window_size
        num_windows = self.original_length // window_size
        x = x.view(B, num_windows, window_size, -1)  # [B, num_windows, window_size, C]
        x = x.view(B, -1, x.shape[-1])  # [B, L, C]
        return x[:, :self.original_length, :]  # 去除填充部分

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [num_windows*B, window_size, C]
        """
        x_norm = self.norm(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)  # Self-Attention
        x = x + self.proj(attn_output)  # 残差连接
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        print(f"self.fc1 = nn.Linear(in_features, hidden_features) {in_features},{hidden_features}")
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        print(f"self.fc2 = nn.Linear(in_features, hidden_features) {in_features},{hidden_features}")
        self.drop = nn.Dropout(drop)
        print(f"self.drop = nn.Dropout(drop) {drop}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer
    包含一个 Multi-Head Self-Attention 和一个 MLP
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super(SwinTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim, drop=drop)

    def forward(self, x):
        # x: [num_windows*B, window_size, C]
        shortcut = x
        x = self.attn(self.norm1(x))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class CustomSwinTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4, window_sizes=[4, 8, 16, 32], num_heads=[8, 8, 16, 16], mlp_ratio=4.0, drop=0.0):
        """
        自定义窗口化 Swin Transformer。
        
        参数:
            embed_dim (int): 嵌入维度。
            num_layers (int): 阶段数，每个阶段对应一个窗口大小。
            window_sizes (list): 每个阶段的窗口大小。
            num_heads (list): 每个阶段的注意力头数。
            mlp_ratio (float): MLP 隐藏层比例。
            drop (float): Dropout 比例。
        """
        super(CustomSwinTransformer, self).__init__()
        assert num_layers == len(window_sizes) == len(num_heads), "num_layers, window_sizes, and num_heads must be the same length."
        
        self.num_layers = num_layers
        self.window_sizes = window_sizes
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        # print("CustomSwinTransformer swin_params['num_heads']:", num_heads)
        # 创建每个阶段的 Swin Transformer 块
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # print("num_heads[i]:",num_heads[i])
            self.layers.append(
                nn.ModuleList([
                    SwinTransformerBlock(embed_dim, num_heads[i], mlp_ratio, drop),
                    # 可以在这里添加更多的模块（如相对位置编码等），根据需要扩展
                ])
            )

    def forward(self, x):
        """
        前向传播。
        
        参数:
            x (Tensor): 输入序列，形状为 [B, L, C]
        
        返回:
            List[Tensor]: 每个阶段的输出，形状为 [B, L, C]
        """
        outputs = []
        for layer_idx, layer in enumerate(self.layers):
            swin_block = layer[0]
            window_size = self.window_sizes[layer_idx]
            # 窗口划分
            partition = WindowPartition(window_size)
            windows = partition(x)  # [num_windows*B, window_size, C]
            
            # 应用 Swin Transformer 块
            windows = swin_block(windows)  # [num_windows*B, window_size, C]
            
            # 窗口逆划分
            reverse = WindowReverse(window_size, self.original_length(x))
            x = reverse(windows)  # [B, L, C]
            
            outputs.append(x)
        return outputs  # List of [B, L, C] from each layer

    def original_length(self, x):
        """
        获取原始序列长度。
        
        参数:
            x (Tensor): 输入序列，形状为 [B, L, C]
        
        返回:
            int: 序列长度 L
        """
        return x.size(1)
