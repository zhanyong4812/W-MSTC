import torch
import torch.nn as nn
from models.feature_fusion.branch1_swin_transformer import CustomSwinTransformer
from models.feature_fusion.branch2_transformer_feature_extractor import TransformerFeatureExtractor
from models.feature_fusion.branch3_cross_attention import CrossAttention

class MultiModalModel(nn.Module):
    def __init__(self, img_dim, iqap_dim, embed_dim, num_heads, window_sizes, swin_params, seq_length, branch=None, block_layers=1):
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

        self.branch = branch if branch else {}
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.window_sizes = window_sizes
        self.block_layers = block_layers  # 堆叠层的数量

        # 输入嵌入层
        self.input_embed_img = nn.Linear(img_dim, embed_dim)
        self.input_embed_iqap = nn.Linear(iqap_dim, embed_dim)

        self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
        
        # 分支1：Swin Transformer
        if self.branch.get("use_branch1", True):
            # print("swin_params['num_heads']:", swin_params['num_heads'])
            self.branch1_swin = CustomSwinTransformer(
                embed_dim=embed_dim,
                num_layers=swin_params['num_layers'],
                window_sizes=swin_params['window_sizes'],
                num_heads=swin_params['num_heads'],
                mlp_ratio=swin_params.get('mlp_ratio', 4.0),
                drop=swin_params.get('drop', 0.0)
            )

        # 分支2：Transformer特征提取器
        if self.branch.get("use_branch2", True):
            self.branch2 = TransformerFeatureExtractor(
                input_dim=embed_dim,
                output_dim=embed_dim,
                seq_length= seq_length,
                window_sizes=window_sizes,
                num_heads=num_heads
            )
            self.downsample_branch2 = nn.Linear(embed_dim, embed_dim)

        # 分支3：跨模态交叉注意力
        if self.branch.get("use_branch3", True):
            self.branch3_cross_attentions = nn.ModuleList([
                CrossAttention(embed_dim=embed_dim, num_heads=num_heads) for _ in range(swin_params['num_layers'])
            ])
            self.downsample_branch3 = nn.Linear(embed_dim * swin_params['num_layers'], embed_dim)

        # 最终全连接层，处理拼接后的输出
        # enabled_branches = sum([self.branch.get(f"use_branch{i}", False) for i in range(1, 4)])
        # self.final_fc = nn.Linear(embed_dim, embed_dim)

    
    def mean_pooling(self, x, target_length):
        """
        将输入张量x的序列长度池化到目标序列长度。

        参数:
            x (tensor): 输入张量，形状为 [batch_size, seq_length, dim]
            target_length (int): 目标序列长度

        返回:
            tensor: 均值池化后的张量，形状为 [batch_size, target_length, dim]
        """
        # print("mean_pooling x input shape",x.shape)
        seq_len = x.shape[1]
        if seq_len == target_length:
            return x  # 如果已经是目标长度，直接返回

        # 计算每个窗口的大小，池化操作需要根据目标长度缩短序列
        step = seq_len // target_length
        pooled = []

        # 对于每个目标长度的元素，进行池化
        for i in range(target_length):
            start_idx = i * step
            end_idx = (i + 1) * step if i != target_length - 1 else seq_len
            pooled.append(x[:, start_idx:end_idx, :].mean(dim=1))  # 在序列维度进行均值池化
        # print(f"torch.stack(pooled, dim=1) shape:{torch.stack(pooled, dim=1).shape}")
        return torch.stack(pooled, dim=1)

    def forward(self, img, iqap):
        # 嵌入输入特征
        x_img = self.input_embed_img(img)
        x_iqap = self.input_embed_iqap(iqap)
        # print(f"x_img shape: {x_img.shape}, x_iqap shape: {x_iqap.shape}")
        # x_img shape: torch.Size([25, 256, 256]), x_iqap shape: torch.Size([25, 256, 256])
        x = torch.cat([x_img, x_iqap], dim=1)
        x = self.mean_pooling(x, self.seq_length)

        # 堆叠块的处理
        for _ in range(self.block_layers):
            # 分支1：Swin Transformer
            mean_branch1_out = None
            if self.branch.get("use_branch1", True):
                branch1_out = self.branch1_swin(x)
                # print(f"branch1_out length:{len(branch1_out)},branch1_out shape: {branch1_out[0].shape}")
                stacked_branch1_out = torch.stack(branch1_out, dim=0)  # 堆叠后的形状： [N, 25, 512, 256]
                mean_branch1_out = torch.mean(stacked_branch1_out, dim=0)  # 在 dim=0 上进行均值池化，输出形状 [25, 512, 256]
                # print(f"mean_branch1_out shape: {mean_branch1_out.shape}")
            # branch1_out[0] shape: torch.Size([25, 512, 256])
            # 分支2：Transformer特征提取器
            branch2_out = None
            if self.branch.get("use_branch2", True):
                branch2_out = self.branch2(x)
                # print(f"branch2_out shape: {branch2_out.shape}")
            # 分支3：跨模态交叉注意力
            branch3_out = None
            if self.branch.get("use_branch3", True):
                branch3_feats = []
                for idx, cross_attn in enumerate(self.branch3_cross_attentions):
                    swin_feat = branch1_out[idx] if idx < len(branch1_out) else torch.zeros(x.size(0), self.embed_dim, device=x.device)
                    cross_attn_out = cross_attn(swin_feat)
                    branch3_feats.append(cross_attn_out)
                branch3_out = torch.stack(branch3_feats) # branch3_out shape: torch.Size([3, 25, 512, 256])
                branch3_out = torch.mean(branch3_out, dim=0)
                # print(f"branch3_out shape: {branch3_out.shape}")
            # 在每个堆叠块结束时将所有分支的输出合并并降维
            final_features = [mean_branch1_out, branch2_out, branch3_out]
            # 打印 final_features 列表中每个元素的形状和类型
            # for i, feat in enumerate(final_features):
            #     print(f"final_features[{i}] shape: {feat.shape}, dtype: {feat.dtype}")
                
            # 只堆叠非 None 的张量
            stacked = torch.stack([feat for feat in final_features if feat is not None], dim=0)

            mean_output = stacked.mean(dim=0)
            # print(f"mean_output stacked shape: {mean_output.shape}")
            # 更新输入特征，继续进行下一个堆叠块的计算
            x = mean_output

        # [batch,length,dim]
        mean_output = x.mean(dim=1)  # [batch, dim]
        # 应用最终的全连接层
        # final_feat = mean_output
        final_feat = self.mlp(mean_output)

        return final_feat
