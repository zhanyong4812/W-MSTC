# models/prototypical_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from ..models.feature_extraction import FeatureExtractorImage, FeatureExtractorIQ
from ..models.feature_fusion import MultiModalModel
from ..config import IMG_DIM, IQAP_DIM, EMBED_DIM, NUM_HEADS, WINDOW_SIZES, SEQ_LENGTH, swin_params, branch, block_layers

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor_img = FeatureExtractorImage()
        self.feature_extractor_iq = FeatureExtractorIQ()
        self.multimodal = MultiModalModel(
            img_dim=IMG_DIM,
            iqap_dim=IQAP_DIM,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            window_sizes=WINDOW_SIZES,
            swin_params=swin_params,
            seq_length=SEQ_LENGTH,
            branch=branch,
            block_layers=block_layers)  
        
    def forward(self, support, queries, support_labels, query_labels):
        """
        support: [batch_size, num_support, channels, height, width]
        queries: [batch_size, num_query, channels, height, width]
        support_labels: [batch_size, num_support]
        query_labels: [batch_size, num_query]
        """
        # print(f"support shape: {support.shape}")
        # print(f"queries shape: {queries.shape}")
        # print(f"support_labels shape: {support_labels.shape}")
        # print(f"query_labels shape: {query_labels.shape}")

        # 提取支持集图像特征
        # print(f" PrototypicalNetwork  input dim support shape:{support.shape}")
        img_features = support[:, :64, :, :, :].reshape(-1, 1, 64, 64)  # 星座图支持集
        # print(f"img_support shape after slicing: {img_features.shape}")
        img_features = self.feature_extractor_img(img_features)  # [N*16, 256]
        
        # 提取支持集 IQ 特征并计算振幅和相位
        iq_raw = support[:, 64:, :, :, :].reshape(-1, 1, 2, 1024)  # IQ分量支持集 [N*16, 1, 2, 1024]
        # print(f"iq_raw_support shape after slicing: {iq_raw.shape}")
        I = iq_raw[:, :, 0, :]  # [N*16, 1, 1024]
        Q = iq_raw[:, :, 1, :]  # [N*16, 1, 1024]
        amplitude = torch.sqrt(I**2 + Q**2).unsqueeze(1)  # [N*16, 1, 1, 1024]
        phase = torch.atan2(Q, I).unsqueeze(1)           # [N*16, 1, 1, 1024]
        I = I.unsqueeze(2)
        Q = Q.unsqueeze(2)
        # 拼接 I, Q, 振幅, 相位，形成 4 通道输入
        iq_processed = torch.cat([I, Q, amplitude, phase], dim=2)  # [N*16, 1, 4, 1024]
        # 使用 torch.permute 交换 dim=1 和 dim=2
        iq_processed = iq_processed.permute(0, 2, 1, 3)  # [N*16, 4, 1, 1024]
        iq_features = self.feature_extractor_iq(iq_processed)     # [N*16, 256]
        
        # 合并图像和 IQ 特征
        # combined_features = torch.cat((img_features, iq_features), dim=1)  # [N*16, 512]
        support_features = self.multimodal(img_features, iq_features)  # [N*16, 256]
        
        # 提取查询集图像特征
        img_features_q = queries[:, :64, :, :, :].reshape(-1, 1, 64, 64)  # 星座图查询集
        img_features_q = self.feature_extractor_img(img_features_q)  # [N*num_query, 256]
        
        # 提取查询集 IQ 特征并计算振幅和相位
        iq_raw_q = queries[:, 64:, :, :, :].reshape(-1, 1, 2, 1024)      # IQ分量查询集 [N*num_query, 1, 2, 1024]
        I_q = iq_raw_q[:, :, 0, :]   # [N*num_query, 1, 1024]
        Q_q = iq_raw_q[:, :, 1, :]  # [N*num_query, 1, 1024]
        amplitude_q = torch.sqrt(I_q**2 + Q_q**2).unsqueeze(1)         # [N*num_query, 1, 1, 1024]
        phase_q = torch.atan2(Q_q, I_q).unsqueeze(1)                  # [N*num_query, 1, 1, 1024]
        I_q = I_q.unsqueeze(2)
        Q_q = Q_q.unsqueeze(2)
        # 拼接 I, Q, 振幅, 相位，形成 4 通道输入
        iq_processed_q = torch.cat([I_q, Q_q, amplitude_q, phase_q], dim=2)  # [N*num_query, 1, 4, 1024]
        iq_processed_q = iq_processed_q.permute(0, 2, 1, 3)  # [N*num_query, 4, 1, 1024]
        iq_features_q = self.feature_extractor_iq(iq_processed_q)              # [N*num_query, 256]
        
        # 合并查询集图像和 IQ 特征
        # combined_features_q = torch.cat((img_features_q, iq_features_q), dim=1)  # [N*num_query, 512]
        query_features = self.multimodal(img_features_q, iq_features_q)  # [N*num_query, 256]
        
        # 计算原型
        prototypes = self.calculate_prototypes(support_features, support_labels)  # [num_classes, 256]
        
        # 计算欧氏距离
        dists = self.euclidean_dist(query_features, prototypes)  # [N*num_query, num_classes]
        
        # 计算对数概率
        log_p_y = F.log_softmax(-dists, dim=1)  # [N*num_query, num_classes]
        
        # 计算损失
        loss = F.nll_loss(log_p_y, query_labels)
        
        # 计算准确率
        acc = (log_p_y.argmax(1) == query_labels).float().mean()
        
        return loss, acc

    def calculate_prototypes(self, features, labels):
        unique_labels = torch.unique(labels)
        prototypes = torch.stack([features[labels == label].mean(0) for label in unique_labels])
        return prototypes

    def euclidean_dist(self, x, y):
        return torch.cdist(x, y, p=2)


def summary_mode():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalNetwork().to(device)
    # for name, module in model.named_modules():
    #     print(name, module)
    support_size = (25, 24, 1, 16, 16)
    query_size = (25, 24, 1, 16, 16)
    support_labels_size = (25,)
    query_labels_size = (25,)
    
    summary(model, input_size=[support_size, query_size, support_labels_size, query_labels_size], depth=10, verbose=True, dtypes=[torch.float32, torch.float32, torch.long, torch.long])