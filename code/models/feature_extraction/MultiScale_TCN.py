import torch
import torch.nn as nn
from torchsummary import summary

class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7], dilation=1):
        super(MultiScaleResidualBlock, self).__init__()
        self.branches = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dilation = dilation

        for k in kernel_sizes:
            left_padding = dilation * (k - 1)
            # Causal padding + convolution + ReLU form one branch
            branch = nn.Sequential(
                nn.ConstantPad2d((left_padding, 0, 0, 0), 0),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, k),
                          padding=0, dilation=(1, dilation)),
                nn.ReLU()
            )
            self.branches.append(branch)

        # Use 1x1 convolution to match residual channels if in/out channels differ
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 1x1 convolution to fuse multi-scale branch outputs
        self.fuse = nn.Conv2d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        # Process each scale's convolution branch in parallel
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate outputs from multiple scales along the channel dimension
        out = torch.cat(branch_outputs, dim=1)
        # Fuse multi-scale features
        out = self.fuse(out)
        # Width alignment
        if out.size(3) != residual.size(3):
            min_width = min(out.size(3), residual.size(3))
            out = out[:, :, :, :min_width]
            residual = residual[:, :, :, :min_width]

        out += residual
        out = self.relu(out)
        return out

class TCN(nn.Module):
    def __init__(self, n_classes):
        super(TCN, self).__init__()
        self.initial_conv = nn.Conv2d(4, 16, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()

        # Replace original single-scale blocks with multi-scale residual blocks
        self.block1 = MultiScaleResidualBlock(16, 32, kernel_sizes=[3,5,7], dilation=1)
        self.block2 = MultiScaleResidualBlock(32, 64, kernel_sizes=[3,5,7], dilation=2)
        self.block3 = MultiScaleResidualBlock(64, 128, kernel_sizes=[3,5,7], dilation=4)
        self.block4 = MultiScaleResidualBlock(128, 256, kernel_sizes=[3,5,7], dilation=8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 256))
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x): 
        # print(f"the shape in TCN first appear is {x.shape}")
        out = self.relu(self.initial_conv(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.squeeze(2)               # [B, C, Seq]
        out = out.transpose(1, 2).contiguous()  # -> [B, Seq, C] consistent with fusion module expectations
        return out

if __name__ == '__main__':
    # Create model instance and print model structure
    n_classes = 10
    model = TCN(n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Print model summary
    # input shape is (channels, height, width)
    summary(model, input_size=(4, 1, 100))
