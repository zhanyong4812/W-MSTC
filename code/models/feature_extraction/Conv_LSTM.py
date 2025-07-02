import torch
import torch.nn as nn
from torchinfo import summary

class SpatialTemporalFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels=1,
                 conv_channels=32,
                 lstm_hidden_size=128,  # 设置隐藏层维度为 128
                 lstm_layers=2,
                 dropout_rate=0.3):
        super(SpatialTemporalFeatureExtractor, self).__init__()

        # 卷积部分：两层卷积块
        self.conv = nn.Sequential(
            # 第1层卷积块
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_rate),

            # 第2层卷积块
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_rate)
        )

        # 单向 LSTM 提取卷积后的时空特征
        # 输入尺寸为卷积输出的通道数（conv_channels * 2）
        self.lstm = nn.LSTM(input_size=conv_channels * 2,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout_rate if lstm_layers > 1 else 0)

    def forward(self, x):
        # 输入 x 的形状: [batch_size, time_steps, height, width]
        # 先在 channel 维度上增加一个维度，变成 [batch_size, time_steps, 1, height, width]
        x = x.unsqueeze(2)
        batch_size, time_steps, channels, height, width = x.size()

        # 合并 batch 和 time_steps 以适应卷积网络：[B*time_steps, channels, height, width]
        x = x.contiguous().view(batch_size * time_steps, channels, height, width)
        x = self.conv(x)  # 输出形状: [B*time_steps, conv_channels*2, H', W']

        # 保持空间结构，转换形状为 [B*time_steps, H', W', conv_channels*2]
        x = x.permute(0, 2, 3, 1)
        N, H_prime, W_prime, C_prime = x.shape

        # 将每个样本的空间位置展平成序列：[B*time_steps, H'*W', C_prime]
        x = x.view(N, H_prime * W_prime, C_prime)

        # 利用 LSTM 处理每个样本的空间位置序列
        # 输入形状: [B*time_steps, sequence_length, input_size]
        lstm_out, _ = self.lstm(x)  # 输出形状: [B*time_steps, sequence_length, lstm_hidden_size]

        # 将时间步信息还原：reshape 为 [batch_size, time_steps, spatial_length, hidden_size]
        spatial_len = lstm_out.shape[1]
        lstm_out = lstm_out.view(batch_size, time_steps, spatial_len, -1)

        # 对时间步维度进行平均聚合，得到每个样本的空间特征
        aggregated = lstm_out.mean(dim=1)  # 结果形状: [batch_size, spatial_length, hidden_size]
        return aggregated

class ConvLSTM(nn.Module):
    def __init__(self, n_classes, lstm_hidden_size=128, dropout_rate=0.3):
        super(ConvLSTM, self).__init__()
        self.feature_extractor = SpatialTemporalFeatureExtractor(
            lstm_hidden_size=lstm_hidden_size,
            dropout_rate=dropout_rate
        )
        # 在特征提取后增加一个 BiLSTM 层，处理“空间序列”（聚合后的每个位置）
        # 此处输入维度为 128，双向后输出维度为 2 * 128 = 256
        self.bi_lstm = nn.LSTM(input_size=lstm_hidden_size,
                               hidden_size=lstm_hidden_size,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=True)
        # 全连接层将 BiLSTM 输出（双向，维度为 256）映射回 128 维
        self.fc = nn.Linear(2 * lstm_hidden_size, lstm_hidden_size)

    def forward(self, x):
        # x 的输入形状: [batch_size, time_steps, height, width]
        features = self.feature_extractor(x)
        # features 的形状: [batch_size, spatial_length, lstm_hidden_size]
        # 使用 BiLSTM 对聚合后的空间序列进行进一步建模
        bi_out, _ = self.bi_lstm(features)  # 输出形状: [batch_size, spatial_length, 2*lstm_hidden_size]
        
        return bi_out

if __name__ == '__main__':
    n_classes = 10
    model = ConvLSTM(n_classes, lstm_hidden_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 构造一个 dummy 输入：[batch_size, time_steps, height, width]
    dummy_input = torch.randn(2, 4, 64, 64).to(device)
    summary(model, input_data=dummy_input, depth=10)
