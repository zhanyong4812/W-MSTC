import torch
import torch.nn as nn
from models.feature_fusion.branch1_swin_transformer import CustomSwinTransformer
from models.feature_fusion.branch2_transformer_feature_extractor import TransformerFeatureExtractor
from models.feature_fusion.branch3_cross_attention import CrossAttention

class MultiModalModel(nn.Module):
    def __init__(self, img_dim, iqap_dim, embed_dim, num_heads, window_sizes,
                 swin_params, seq_length, branch=None, block_layers=1):
        """
        初始化多模态模型。

        参数:
            img_dim (int): IMG模态的维度。
            iqap_dim (int): IQAP模态的维度。
            embed_dim (int): 嵌入维度。
            num_heads (int): 自注意力的头数。
            window_sizes (list): 不同窗口大小的列表。
            swin_params (dict): CustomSwinTransformer的参数。
            seq_length (int): 序列长度。
            branch (dict): 用于控制分支使用的配置文件。
            block_layers (int): 堆叠块的层数。
        """
        super(MultiModalModel, self).__init__()

        self.branch = branch or {}
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.window_sizes = window_sizes
        self.block_layers = block_layers

        # 输入嵌入层
        self.input_embed_img  = nn.Linear(img_dim,  embed_dim)
        self.input_embed_iqap = nn.Linear(iqap_dim, embed_dim)

        # 最终 MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # 分支1：Swin Transformer
        if self.branch.get("use_branch1", True):
            self.branch1_swin = CustomSwinTransformer(
                embed_dim=embed_dim,
                num_layers=swin_params['num_layers'],
                window_sizes=swin_params['window_sizes'],
                num_heads=swin_params['num_heads'],
                mlp_ratio=swin_params.get('mlp_ratio', 4.0),
                drop=swin_params.get('drop', 0.0)
            )

        # 分支2：Transformer 特征提取器
        if self.branch.get("use_branch2", True):
            self.branch2 = TransformerFeatureExtractor(
                input_dim=embed_dim,
                output_dim=embed_dim,
                seq_length=seq_length,
                window_sizes=window_sizes,
                num_heads=num_heads
            )

        # 分支3：跨模态交叉注意力
        if self.branch.get("use_branch3", True):
            self.branch3_cross_attentions = nn.ModuleList([
                CrossAttention(embed_dim=embed_dim, num_heads=num_heads)
                for _ in range(swin_params['num_layers'])
            ])

        # 动态统计启用的分支数
        self.enabled_branches = []
        if self.branch.get("use_branch1", True): self.enabled_branches.append("branch1")
        if self.branch.get("use_branch2", True): self.enabled_branches.append("branch2")
        if self.branch.get("use_branch3", True): self.enabled_branches.append("branch3")
        self.num_branches = len(self.enabled_branches)

        # 可学习的门控参数 θ（长度 = 启用的分支数）
        self.theta = nn.Parameter(torch.zeros(self.num_branches))

    def mean_pooling(self, x, target_length):
        """
        均值池化到目标序列长度
        """
        seq_len = x.shape[1]
        if seq_len == target_length:
            return x

        step = seq_len // target_length
        pooled = []
        for i in range(target_length):
            start = i * step
            end   = (i + 1) * step if i != target_length - 1 else seq_len
            pooled.append(x[:, start:end, :].mean(dim=1))
        return torch.stack(pooled, dim=1)

    def forward(self, img, iqap):
        # 嵌入
        x_img  = self.input_embed_img(img)
        x_iqap = self.input_embed_iqap(iqap)
        x = torch.cat([x_img, x_iqap], dim=1)               # [batch, 2*seq, embed_dim]
        x = self.mean_pooling(x, self.seq_length)           # [batch, seq, embed_dim]

        # 多个堆叠块
        for _ in range(self.block_layers):
            # --- 分支1 输出 ---
            mean_branch1_out = None
            if self.branch.get("use_branch1", True):
                branch1_out = self.branch1_swin(x)          # list of [batch, seq, embed_dim]
                stacked1    = torch.stack(branch1_out, dim=0) 
                mean_branch1_out = stacked1.mean(dim=0)     # [batch, seq, embed_dim]

            # --- 分支2 输出 ---
            branch2_out = None
            if self.branch.get("use_branch2", True):
                branch2_out = self.branch2(x)               # [batch, seq, embed_dim]

            # --- 分支3 输出 ---
            branch3_out = None
            if self.branch.get("use_branch3", True):
                feats3 = []
                for idx, cross_attn in enumerate(self.branch3_cross_attentions):
                    swin_feat = branch1_out[idx] if idx < len(branch1_out) else torch.zeros_like(x)
                    feats3.append(cross_attn(swin_feat))    # [batch, seq, embed_dim]
                stacked3   = torch.stack(feats3, dim=0)
                branch3_out = stacked3.mean(dim=0)        # [batch, seq, embed_dim]

            # 收集所有分支特征
            final_features = [mean_branch1_out, branch2_out, branch3_out]
            feats = [f for f in final_features if f is not None]  # 数量 = self.num_branches

            # 可学习门控融合
            stacked = torch.stack(feats, dim=0)                 # [B, batch, seq, embed_dim]
            weights = torch.softmax(self.theta, dim=0)          # [B]
            weights = weights.view(self.num_branches, 1, 1, 1)
            x = (stacked * weights).sum(dim=0)                  # [batch, seq, embed_dim]

        # 池化并通过 MLP
        out = x.mean(dim=1)                                     # [batch, embed_dim]
        out = self.mlp(out)                                     # [batch, embed_dim]
        return out
