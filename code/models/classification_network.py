import torch
import torch.nn as nn
from models.feature_extraction.Conv_LSTM import ConvLSTM
from models.feature_extraction.MultiScale_TCN import TCN
from models.feature_fusion.multimodal_model import MultiModalModel
from config import (
    IMG_DIM, IQAP_DIM, EMBED_DIM, NUM_CLASSES,
    NUM_HEADS, WINDOW_SIZES, swin_params,
    SEQ_LENGTH, branch, block_layers
)

class ClassificationNetwork(nn.Module):
    def __init__(self):
        """
        使用原有的 ConvLSTM（空时特征提取）、TCN（多尺度时序特征）、
        以及 MultiModalModel 融合模块，新增一个全连接分类头。
        """
        super(ClassificationNetwork, self).__init__()

        # ———— 1. 图像分支特征提取器（ConvLSTM） ————
        # ConvLSTM 的构造参数可按需求调整。此处 n_classes=None，不用于分类，仅输出特征。
        self.feature_extractor_img = ConvLSTM(n_classes=NUM_CLASSES)

        # ———— 2. IQ 分支特征提取器（TCN） ————
        self.feature_extractor_iq = TCN(n_classes=NUM_CLASSES)

        # ———— 3. 特征融合模块（MultiModalModel） ————
        # 参数说明：
        #   img_dim    = ConvLSTM 输出的特征维度（例如隐藏态大小 = IMG_DIM）
        #   iqap_dim   = TCN 输出的特征维度（例如 128 = IQAP_DIM）
        #   embed_dim  = 融合后输出维度（EMBED_DIM）
        self.multimodal = MultiModalModel(
            img_dim      = IMG_DIM,
            iqap_dim     = IQAP_DIM,
            embed_dim    = EMBED_DIM,
            num_heads    = NUM_HEADS,
            window_sizes = WINDOW_SIZES,
            swin_params  = swin_params,
            seq_length   = SEQ_LENGTH,
            branch       = branch,
            block_layers = block_layers
        )

        # ———— 4. 分类头：把融合后 EMBED_DIM 维特征映射到 NUM_CLASSES 维 logits ————
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, img_batch, iq_batch):
        """
        img_batch: Tensor, 形状 [B, IMG_STACK, H, W]
                   —— 与 ConvLSTM 要求的输入 [batch, time_steps, height, width] 保持一致。
        iq_batch:  Tensor, 形状 [B, IQ_STACK, 1, L]
                   —— 与 TCN 输入 [batch, channels, 1, length] 保持一致。

        返回:
            logits: Tensor, 形状 [B, NUM_CLASSES]
        """
        # ———— 1. 提取图像分支特征 ————
        # 假设 ConvLSTM.forward(img_batch) 输出形状 [B, seq_len_img, IMG_DIM]
        img_features = self.feature_extractor_img(img_batch)
        # print(f"img_features shape after ConvLSTM: {img_features.shape}")
        # 如果实际输出不正是 [B, seq_len_img, IMG_DIM]，可在这里做 reshape 或 mean 操作：
        # img_features = img_features.mean(dim=1, keepdim=False)

        # ———— 2. 提取 IQ 分支特征 ————
        # 假设 TCN.forward(iq_batch) 输出形状 [B, seq_len_iq, IQAP_DIM]
        # 2.1 原始 I/Q
        # iq_batch[:, 0, 0, :] → I  (shape [B, 1024])
        # iq_batch[:, 1, 0, :] → Q  (shape [B, 1024])
        # print(f"iq_batch shape: {iq_batch.shape}")
        I = iq_batch[:, 0, 0, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 1024]
        Q = iq_batch[:, 1, 0, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 1024]

        # 2.2 计算振幅与相位
        amplitude = torch.sqrt(I ** 2 + Q ** 2)             # [B, 1, 1, 1024]
        phase     = torch.atan2(Q, I)                       # [B, 1, 1, 1024]

        # 2.3 将 I, Q, 振幅, 相位 按“通道”拼接 → 四通道
        # cat 维度=1：从 [B,1,1,1024] × 4 → [B,4,1,1024]
        iqap = torch.cat([I, Q, amplitude, phase], dim=1)   # [B, 4, 1, 1024]
        # print(f"iqap shape after concatenation: {iqap.shape}")
        iq_features = self.feature_extractor_iq(iqap)
        # print(f"iq_features shape after TCN: {iq_features.shape}")
        # 同样可视需要做 reshape 或 unsqueeze，使其符合 MultiModalModel 要求。

        # ———— 3. 多模态特征融合 ————
        # 融合后 fused_features: [B, EMBED_DIM]
        fused_features = self.multimodal(img_features, iq_features)
        # print(f"fused_features shape: {fused_features.shape}")
        # 如果 fused_features 的形状是 [B, seq_len_fuse, EMBED_DIM]，请改为：
        # fused_features = fused_features.mean(dim=1, keepdim=False)

        # ———— 4. 分类头 → logits ————
        logits = self.classifier(fused_features)  # [B, NUM_CLASSES]
        return logits
