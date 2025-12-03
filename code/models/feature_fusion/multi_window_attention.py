import torch
import torch.nn as nn


class DMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super(DMLP, self).__init__()
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
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)
        self.mlp = DMLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            drop=drop,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_length, feature_dim]
        """
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)  # Self-Attention
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))  # MLP residual
        return x


class MultiWindowAttention(nn.Module):
    """
    MWA: Multi-Window Attention (DMLP + LN + MHAL)
    """

    def __init__(self, input_dim=256, output_dim=256, seq_length=512, window_sizes=[32, 64, 128, 256, 512], num_heads=8):
        super(MultiWindowAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.window_sizes = window_sizes
        self.num_heads = num_heads

        self.local_attentions = nn.ModuleList(
            [LocalSelfAttention(embed_dim=input_dim, heads=num_heads) for _ in window_sizes]
        )

        self.fc = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in window_sizes])
        self.final_fc = nn.Linear(len(window_sizes) * output_dim, output_dim)

    def forward(self, x):
        """
        x: [N, S, D]
        """
        branch_outputs = []

        for idx, window_size in enumerate(self.window_sizes):
            step = window_size
            if window_size > self.seq_length:
                raise ValueError(f"Window size {window_size} is larger than sequence length {self.seq_length}")

            num_windows = (self.seq_length - window_size) // step + 1
            windows = x.unfold(1, window_size, step)  # [N, num_windows, window_size, D]

            if window_size == self.seq_length:
                num_windows = 1
                windows = x.unsqueeze(1)  # [N, 1, S, D]

            windows = windows.contiguous().view(-1, window_size, self.input_dim)  # [N * num_windows, window_size, D]

            windows = self.local_attentions[idx](windows)
            windows = self.fc[idx](windows)

            windows = windows.view(x.size(0), num_windows, window_size, self.output_dim).contiguous()
            windows_agg = windows.view(-1, self.seq_length, self.output_dim)

            branch_outputs.append(windows_agg)

        final_output = torch.mean(torch.stack(branch_outputs), dim=0)
        return final_output

