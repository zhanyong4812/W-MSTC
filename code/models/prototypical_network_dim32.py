# models/prototypical_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from models.feature_extraction import TCN, ConvLSTM
from models.feature_fusion import MultiModalModel
from config import *

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor_img = ConvLSTM(n_classes=N_WAY)
        self.feature_extractor_iq = TCN(n_classes=N_WAY)
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
        本次修改内容如下：
        - (24, 1000, 66, 1, 32, 32) -> (24, 1000, 64, 1, 32, 32) + (24, 1000, 2, 1, 32, 32)
        - (24, 1000, 64, 1, 32, 32) -> (24, 1000, 16, 64, 64)
        - (24, 1000, 2, 1, 32, 32) -> (24, 1000, 2, 1, 1024)
        """
        # 提取支持集图像特征
        # print(f" PrototypicalNetwork  input dim support shape:{support.shape}")
        img_features = support[:, :STACK_IMG_SIZE, :, :, :].reshape(-1, STACK_IMG_SIZE, RESIZE_WIDTH, RESIZE_WIDTH)  # 星座图支持集
        # print(f"img_support shape after slicing: {img_features.shape}")
        img_features = self.feature_extractor_img(img_features)  # [N*64, 256]
        # print(f"img_features shape: {img_features.shape}")
        # 提取支持集 IQ 特征并计算振幅和相位
        iq_raw = support[:, STACK_IMG_SIZE:, :, :, :].reshape(-1, 1, 2, 1024)  # IQ分量支持集 [N*64, 1, 2, 1024]
        # print(f"iq_raw_support shape after slicing: {iq_raw.shape}")
        I = iq_raw[:, :, 0, :]  # [N*64, 1, 1024]
        Q = iq_raw[:, :, 1, :]  # [N*64, 1, 1024]
        amplitude = torch.sqrt(I**2 + Q**2).unsqueeze(1)  # [N*64, 1, 1, 1024]
        phase = torch.atan2(Q, I).unsqueeze(1)           # [N*64, 1, 1, 1024]
        I = I.unsqueeze(2)
        Q = Q.unsqueeze(2)
        # 拼接 I, Q, 振幅, 相位，形成 STACK_IMG_SIZE 通道输入
        iq_processed = torch.cat([I, Q, amplitude, phase], dim=2)  # [N*64, 1, STACK_IMG_SIZE, 1024]
        # 使用 torch.permute 交换 dim=1 和 dim=2
        iq_processed = iq_processed.permute(0, 2, 1, 3)  # [N*64, STACK_IMG_SIZE, 1, 1024]
        # print(f"before feature_extractor_iq iq_processed shape:{iq_processed.shape}")
        iq_features = self.feature_extractor_iq(iq_processed)     # [N*64, 256]
        # print(f"iq_features shape: {iq_features.shape}") # iq_features shape: torch.Size([625, 256, 256])
        #### IQ none
        # img_features = torch.zeros_like(img_features)  # [N*64, 256]
        
        # 合并图像和 IQ 特征
        # combined_features = torch.cat((img_features, iq_features), dim=1)  # [N*64, 512]
        # print(f"img_features is {img_features.shape},iq_features is {iq_features.shape} ")
        support_features = self.multimodal(img_features, iq_features)  # [N*64, 256]
        
        # 提取查询集图像特征
        img_features_q = queries[:, :STACK_IMG_SIZE, :, :, :].reshape(-1, STACK_IMG_SIZE, RESIZE_WIDTH, RESIZE_WIDTH)  # 星座图查询集
        img_features_q = self.feature_extractor_img(img_features_q)  # [N*num_query, 256]
        
        # 提取查询集 IQ 特征并计算振幅和相位
        iq_raw_q = queries[:, STACK_IMG_SIZE:, :, :, :].reshape(-1, 1, 2, 1024)      # IQ分量查询集 [N*num_query, 1, 2, 1024]
        I_q = iq_raw_q[:, :, 0, :]   # [N*num_query, 1, 1024]
        Q_q = iq_raw_q[:, :, 1, :]  # [N*num_query, 1, 1024]
        amplitude_q = torch.sqrt(I_q**2 + Q_q**2).unsqueeze(1)         # [N*num_query, 1, 1, 1024]
        phase_q = torch.atan2(Q_q, I_q).unsqueeze(1)                  # [N*num_query, 1, 1, 1024]
        I_q = I_q.unsqueeze(2)
        Q_q = Q_q.unsqueeze(2)
        # 拼接 I, Q, 振幅, 相位，形成 STACK_IMG_SIZE 通道输入
        iq_processed_q = torch.cat([I_q, Q_q, amplitude_q, phase_q], dim=2)  # [N*num_query, 1, STACK_IMG_SIZE, 1024]
        iq_processed_q = iq_processed_q.permute(0, 2, 1, 3)  # [N*num_query, STACK_IMG_SIZE, 1, 1024]
        iq_features_q = self.feature_extractor_iq(iq_processed_q)              # [N*num_query, 256]
        
        # 合并查询集图像和 IQ 特征
        # combined_features_q = torch.cat((img_features_q, iq_features_q), dim=1)  # [N*num_query, 512]
        #### IQ_q none
        ### 
        # img_features_q = torch.zeros_like(img_features_q)  # [N*64, 256]
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
    support_size = (15, 10, 1, 32, 32)
    query_size = (15, 10, 1,32, 32)
    support_labels_size = (15,)
    query_labels_size = (15,)
    
    summary(model, input_size=[support_size, query_size, support_labels_size, query_labels_size], depth=10, verbose=True, dtypes=[torch.float32, torch.float32, torch.long, torch.long])

if __name__ == "__main__":
    summary_mode()