# branch2_transformer_feature_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class LocalSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, mlp_ratio=4, drop=0.0):
        super(LocalSelfAttention, self).__init__()
        
        # Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)
        
        # MLP部分
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim, drop=drop)
        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_length, feature_dim]
        """
        # 先对输入进行LayerNorm
        x_norm = self.norm1(x)
        # Self-Attention计算
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)  # Self-Attention
        # 残差连接：attn_output 和原始输入加在一起
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))  # MLP的残差连接
        return x


class TransformerFeatureExtractor(nn.Module):
    # need dropout 0.2
    def __init__(self, input_dim=256, output_dim=256, seq_length=512, window_sizes=[32, 64, 128, 256, 512], num_heads=8):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.window_sizes = window_sizes  # 多个窗口大小
        self.num_heads = num_heads
        
        # 为每个窗口大小创建一个局部自注意力层
        self.local_attentions = nn.ModuleList([
            LocalSelfAttention(embed_dim=input_dim, heads=num_heads) for _ in window_sizes
        ])
        
        # 全连接层，用于特征变换
        self.fc = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in window_sizes
        ])
        
        # 最终的全连接降维层，将所有窗口大小的输出合并为一个输出
        self.final_fc = nn.Linear(len(window_sizes) * output_dim, output_dim)
    
    def forward(self, x):
        """
        x: [N, S, D] 输入张量
        """
        # print(f"branch2 TransformerFeatureExtractor input shape: {x.shape}")
        # branch2 before multi-scale attention shape torch.Size([25, 512, 256])
        branch_outputs = []
        
        for idx, window_size in enumerate(self.window_sizes):
            # 确定步长，这里设置为窗口大小，确保不重叠
            step = window_size
            # print(f"window_size: {window_size}")
            if window_size > self.seq_length:
                raise ValueError(f"窗口大小 {window_size} 大于序列长度 {self.seq_length}")
            # print(f"self.seq_length: {self.seq_length}")
            # 计算窗口数量
            num_windows = (self.seq_length - window_size) // step + 1
            # print(f"num_windows: {num_windows}")
            # 利用滑动窗口切分数据
            windows = x.unfold(1, window_size, step)  # [N, num_windows, window_size, D]
            
            # 如果窗口大小等于序列长度（512），则 num_windows=1
            if window_size != self.seq_length:
                pass  # 不需要调整维度顺序
            else:
                # 对于全局窗口，不进行切分，只保留原始序列
                num_windows = 1
                windows = x.unsqueeze(1)  # [N, 1, S, D]
            
            # 将窗口展平成批次维度的一部分，以便批量处理
            windows = windows.contiguous().view(-1, window_size, self.input_dim)  # [N * num_windows, window_size, D]
            
            # 对每个窗口应用局部自注意力
            windows = self.local_attentions[idx](windows)  # [N * num_windows, window_size, D]
            
            # 通过全连接层进行特征变换
            windows = self.fc[idx](windows)  # [N * num_windows, window_size, D]
            
            # print("window_size:" ,window_size)
            # print("branch2 TransformerFeatureExtractor after windows dense shape: ", windows.shape)
            # window_size: 32
            # branch2 TransformerFeatureExtractor after windows dense shape:  torch.Size([400, 32, 256])
            # 400 = 25 * 16  16 * 32 = 512
            # 将处理后的窗口恢复到原来的形状 [N, num_windows, window_size, D]
            # print(f"x.size(0):{x.size(0)}, num_windows:{num_windows}, window_size:{window_size}, self.output_dim:{self.output_dim}")
            windows = windows.view(x.size(0), num_windows, window_size, self.output_dim).contiguous()  # [N, num_windows, window_size, D]
            # print(f"branch2 inner windows shape {windows.shape}")
            # 需要变换回原始的数据形状即[-1, self.seq_length * 2, output_dim]
            # window_mean = windows.mean(dim=2)  # [N, num_windows, D]
            # 聚合所有窗口的表示，例如取平均
            # windows_agg = window_mean.mean(dim=1)  # [N, D]
            # 收集每个窗口大小的聚合输出
            windows_agg = windows.view(-1, self.seq_length, self.output_dim)  # [-1, seq_length * 2, output_dim]
    
            branch_outputs.append(windows_agg)
        """
        branch2 inner windows shape torch.Size([25, 16, 32, 256])
        branch2 inner windows shape torch.Size([25, 4, 128, 256])
        branch2 inner windows shape torch.Size([25, 1, 512, 256])
        branch2 inner length: 3, windows shape:torch.Size([25, 256])
        """
        # print(f"branch2 inner length: {len(branch_outputs)}, windows shape:{branch_outputs[0].shape}")
        # 将所有窗口大小的输出拼接起来
        final_output = torch.mean(torch.stack(branch_outputs), dim=0)  # [N, seq_length * 2, output_dim]
        # print(f"branch2 inner final_output shape: {final_output.shape}")     
        return final_output
