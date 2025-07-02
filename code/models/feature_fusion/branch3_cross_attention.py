import torch
import torch.nn as nn


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
        
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4 , split_ratio=0.5, drop=0.0):
        """
        split_ratio: 分割比例，用于确定 S_A 和 S_B 的长度
        mlp_ratio: MLP部分的比例，默认是 4.0，表示映射到 4 倍的维度
        """
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim, drop=drop)
        self.activation = nn.GELU()
        self.split_ratio = split_ratio

    def forward(self, x):
        """
        x: [B, S, C]，其中前 S_A = S * split_ratio 是模态 A，后 S_B = S * (1 - split_ratio) 是模态 B
        """
        B, S, C = x.shape
        S_A = int(S * self.split_ratio)
        S_B = S - S_A

        # 拆分模态 A 和模态 B
        query_A = x[:, :S_A, :]  # [B, S_A, C]
        key_B = x[:, S_A:, :]    # [B, S_B, C]
        value_B = x[:, S_A:, :]  # [B, S_B, C]

        # 归一化
        q_A = self.norm1(query_A)  # [B, S_A, C]
        k_B = self.norm1(key_B)    # [B, S_B, C]
        v_B = self.norm1(value_B)  # [B, S_B, C]
        
        # 计算跨模态注意力: B A A 模式
        attn_output_A = self.cross_attention(q_A, k_B, v_B)[0]  # [B, S_A, C]
        
        # 投影和激活
        attn_output_A = self.proj(attn_output_A)  # [B, S_A, C]
        attn_output_A = self.activation(attn_output_A)  # [B, S_A, C]

        # 残差连接: A 模态
        attn_output_A = x[:, :S_A, :] + attn_output_A  # [B, S_A, C]

        # 计算跨模态注意力: A B B 模式
        query_B = x[:, S_A:, :]  # [B, S_B, C]
        key_A = x[:, :S_A, :]    # [B, S_A, C]
        value_A = x[:, :S_A, :]  # [B, S_A, C]

        q_B = self.norm1(query_B)  # [B, S_B, C]
        k_A = self.norm1(key_A)    # [B, S_A, C]
        v_A = self.norm1(value_A)  # [B, S_A, C]
        
        # 计算跨模态注意力: A B B 模式
        attn_output_B = self.cross_attention(q_B, k_A, v_A)[0]  # [B, S_B, C]
        
        # 投影和激活
        attn_output_B = self.proj(attn_output_B)  # [B, S_B, C]
        attn_output_B = self.activation(attn_output_B)  # [B, S_B, C]

        # 残差连接: B 模态
        attn_output_B = x[:, S_A:, :] + attn_output_B  # [B, S_B, C]

        # 拼接 A B A 和 A B B 模式的输出
        output = torch.cat([attn_output_A, attn_output_B], dim=1)  # [B, S, C]，S 是拼接后的长度
        # 残差连接
        output = output + self.mlp(self.norm2(output))
        return output
