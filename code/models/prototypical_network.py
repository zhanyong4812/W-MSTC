# models/prototypical_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from models.feature_extraction import TCN, ConvLSTM
from models.feature_fusion import WindowedCrossAttention
from config import *

class PrototypicalNetwork(nn.Module):
    def __init__(self):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor_img = ConvLSTM(n_classes=N_WAY)
        self.feature_extractor_iq = TCN(n_classes=N_WAY)
        self.multimodal = WindowedCrossAttention(
            img_dim=IMG_DIM,
            iqap_dim=IQAP_DIM,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            window_sizes=WINDOW_SIZES,
            swin_params=swin_params,
            seq_length=SEQ_LENGTH,
            module_flags=branch,
            block_layers=block_layers)  
        
    def forward(self, support, queries, support_labels, query_labels):
        img_features = support[:, :STACK_IMG_SIZE, :, :, :].reshape(-1, STACK_IMG_SIZE, RESIZE_WIDTH, RESIZE_WIDTH)
        img_features = self.feature_extractor_img(img_features)
        iq_spatial = support[:, STACK_IMG_SIZE:, :, :, :]
        batch_size = iq_spatial.shape[0]
        iq_flat = iq_spatial.reshape(batch_size, 2, -1)
        iq_raw = iq_flat.unsqueeze(1)
        I = iq_raw[:, :, 0, :]
        Q = iq_raw[:, :, 1, :]
        amplitude = torch.sqrt(I**2 + Q**2).unsqueeze(1)
        phase = torch.atan2(Q, I).unsqueeze(1)
        I = I.unsqueeze(2)
        Q = Q.unsqueeze(2)
        iq_processed = torch.cat([I, Q, amplitude, phase], dim=2)
        iq_processed = iq_processed.permute(0, 2, 1, 3)
        iq_features = self.feature_extractor_iq(iq_processed)
        support_features = self.multimodal(img_features, iq_features)

        img_features_q = queries[:, :STACK_IMG_SIZE, :, :, :].reshape(-1, STACK_IMG_SIZE, RESIZE_WIDTH, RESIZE_WIDTH)
        img_features_q = self.feature_extractor_img(img_features_q)
        iq_spatial_q = queries[:, STACK_IMG_SIZE:, :, :, :]
        batch_size_q = iq_spatial_q.shape[0]
        iq_flat_q = iq_spatial_q.reshape(batch_size_q, 2, -1)
        iq_raw_q = iq_flat_q.unsqueeze(1)
        I_q = iq_raw_q[:, :, 0, :]
        Q_q = iq_raw_q[:, :, 1, :]
        amplitude_q = torch.sqrt(I_q**2 + Q_q**2).unsqueeze(1)
        phase_q = torch.atan2(Q_q, I_q).unsqueeze(1)
        I_q = I_q.unsqueeze(2)
        Q_q = Q_q.unsqueeze(2)
        iq_processed_q = torch.cat([I_q, Q_q, amplitude_q, phase_q], dim=2)
        iq_processed_q = iq_processed_q.permute(0, 2, 1, 3)
        iq_features_q = self.feature_extractor_iq(iq_processed_q)
        query_features = self.multimodal(img_features_q, iq_features_q)
        prototypes = self.calculate_prototypes(support_features, support_labels)
        dists = self.euclidean_dist(query_features, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_p_y, query_labels)
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
    support_size = (15, 10, 1, 32, 32)
    query_size = (15, 10, 1,32, 32)
    support_labels_size = (15,)
    query_labels_size = (15,)
    
    summary(model, input_size=[support_size, query_size, support_labels_size, query_labels_size], depth=10, verbose=True, dtypes=[torch.float32, torch.float32, torch.long, torch.long])

if __name__ == "__main__":
    summary_mode()
