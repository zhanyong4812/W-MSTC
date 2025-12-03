import torch
import torch.nn as nn

from .feature_extraction import ConvLSTM, TCN
from .feature_fusion import WindowedCrossAttention
from config import (
    IMG_DIM,
    IQAP_DIM,
    EMBED_DIM,
    NUM_HEADS,
    WINDOW_SIZES,
    swin_params,
    SEQ_LENGTH,
    branch,
    block_layers,
    CLS_NUM_CLASSES,
)


class ClassificationNetwork(nn.Module):
    """
    Non-few-shot supervised classification network:
      - Image branch: ConvLSTM spatiotemporal feature extraction
      - IQ branch: MultiScale TCN temporal feature extraction
      - Fusion: Windowed Cross-Attention (WA + MWA + CA)
      - Classification head: Fully connected to CLS_NUM_CLASSES-dimensional logits
    """

    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        # 1. Image branch feature extractor (ConvLSTM)
        # The n_classes here is just a placeholder and doesn't participate in the actual classification head; ConvLSTM is used as a feature extractor
        self.feature_extractor_img = ConvLSTM(n_classes=CLS_NUM_CLASSES)

        # 2. IQ branch feature extractor (TCN)
        self.feature_extractor_iq = TCN(n_classes=CLS_NUM_CLASSES)

        # 3. Feature fusion module (Windowed Cross-Attention: WA + MWA + CA)
        self.multimodal = WindowedCrossAttention(
            img_dim=IMG_DIM,
            iqap_dim=IQAP_DIM,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            window_sizes=WINDOW_SIZES,
            swin_params=swin_params,
            seq_length=SEQ_LENGTH,
            module_flags=branch,
            block_layers=block_layers,
        )

        # 4. Classification head: Fused EMBED_DIM → CLS_NUM_CLASSES
        self.classifier = nn.Linear(EMBED_DIM, CLS_NUM_CLASSES)

    def forward(self, img_batch, iq_batch):
        """
        img_batch: [B, IMG_STACK, H, W]
        iq_batch:  [B, IQ_STACK, 1, L]  (Will split I/Q and construct 4-channel IQAP later)
        """
        # 1. Image branch features
        img_features = self.feature_extractor_img(img_batch)

        # 2. IQ branch features: First recover I/Q from stacked IQ images, then calculate amplitude and phase
        # Assume the first two channels of iq_batch correspond to I/Q respectively:
        # iq_batch[:, 0, 0, :] → I, iq_batch[:, 1, 0, :] → Q
        I = iq_batch[:, 0, 0, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        Q = iq_batch[:, 1, 0, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

        amplitude = torch.sqrt(I ** 2 + Q ** 2)
        phase = torch.atan2(Q, I)

        # Concatenate I/Q/Amplitude/Phase by "channel" → [B, 4, 1, L]
        iqap = torch.cat([I, Q, amplitude, phase], dim=1)
        iq_features = self.feature_extractor_iq(iqap)  # [B, seq_iq, IQAP_DIM]

        # 3. Multimodal feature fusion
        fused_features = self.multimodal(img_features, iq_features)  # [B, EMBED_DIM]

        # 4. Classification head
        logits = self.classifier(fused_features)
        return logits


