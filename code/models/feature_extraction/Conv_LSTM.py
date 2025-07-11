import torch
import torch.nn as nn
from torchinfo import summary

class SpatialTemporalFeatureExtractor(nn.Module):
    def __init__(self,
                 in_channels=1,
                 conv_channels=32,
                 lstm_hidden_size=128,
                 lstm_layers=2,
                 dropout_rate=0.3):
        super(SpatialTemporalFeatureExtractor, self).__init__()

        # Convolutional part: two convolutional blocks
        self.conv = nn.Sequential(
            # 1st convolutional block
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_rate),

            # 2nd convolutional block
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_rate)
        )

        # Unidirectional LSTM to extract spatio-temporal features after convolution
        # Input size is number of channels after conv: conv_channels * 2
        self.lstm = nn.LSTM(input_size=conv_channels * 2,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout_rate if lstm_layers > 1 else 0)

    def forward(self, x):
        # Input x shape: [batch_size, time_steps, height, width]
        # Add channel dimension: [batch_size, time_steps, 1, height, width]
        x = x.unsqueeze(2)
        batch_size, time_steps, channels, height, width = x.size()

        # Merge batch and time_steps to apply convolution: [B*time_steps, channels, height, width]
        x = x.contiguous().view(batch_size * time_steps, channels, height, width)
        x = self.conv(x)  # Output shape: [B*time_steps, conv_channels*2, H', W']

        # Preserve spatial structure: [B*time_steps, H', W', conv_channels*2]
        x = x.permute(0, 2, 3, 1)
        N, H_prime, W_prime, C_prime = x.shape

        # Flatten spatial positions into a sequence: [B*time_steps, H'*W', C_prime]
        x = x.view(N, H_prime * W_prime, C_prime)

        # LSTM over the spatial sequence for each sample
        # Input shape: [B*time_steps, sequence_length, input_size]
        lstm_out, _ = self.lstm(x)  # Output shape: [B*time_steps, sequence_length, lstm_hidden_size]

        # Restore time_steps dimension: [batch_size, time_steps, spatial_length, hidden_size]
        spatial_len = lstm_out.shape[1]
        lstm_out = lstm_out.view(batch_size, time_steps, spatial_len, -1)

        # Average over time_steps dimension to get per-sample spatial features
        aggregated = lstm_out.mean(dim=1)  # Result shape: [batch_size, spatial_length, hidden_size]
        return aggregated

class ConvLSTM(nn.Module):
    def __init__(self, n_classes, lstm_hidden_size=128, dropout_rate=0.3):
        super(ConvLSTM, self).__init__()
        self.feature_extractor = SpatialTemporalFeatureExtractor(
            lstm_hidden_size=lstm_hidden_size,
            dropout_rate=dropout_rate
        )
        # Add a BiLSTM layer after feature extraction to process the 'spatial sequence'
        # Input dimension is lstm_hidden_size; bi-directional output is 2 * lstm_hidden_size
        self.bi_lstm = nn.LSTM(input_size=lstm_hidden_size,
                               hidden_size=lstm_hidden_size,
                               num_layers=2,
                               batch_first=True,
                               bidirectional=True)
        # Fully connected layer to map BiLSTM output (2*lstm_hidden_size) back to lstm_hidden_size
        self.fc = nn.Linear(2 * lstm_hidden_size, lstm_hidden_size)

    def forward(self, x):
        # Input x shape: [batch_size, time_steps, height, width]
        features = self.feature_extractor(x)
        # features shape: [batch_size, spatial_length, lstm_hidden_size]
        # Apply BiLSTM over the aggregated spatial sequence
        bi_out, _ = self.bi_lstm(features)  # Output shape: [batch_size, spatial_length, 2*lstm_hidden_size]
        return bi_out

if __name__ == '__main__':
    n_classes = 10
    model = ConvLSTM(n_classes, lstm_hidden_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create a dummy input: [batch_size, time_steps, height, width]
    dummy_input = torch.randn(2, 4, 64, 64).to(device)
    summary(model, input_data=dummy_input, depth=10)
