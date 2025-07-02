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
            # 因果填充 + 卷积 + ReLU组成一个分支
            branch = nn.Sequential(
                nn.ConstantPad2d((left_padding, 0, 0, 0), 0),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, k),
                          padding=0, dilation=(1, dilation)),
                nn.ReLU()
            )
            self.branches.append(branch)

        # 如果输入和输出通道数不同，则使用1x1卷积调整残差通道数
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 融合多尺度分支输出的1x1卷积
        self.fuse = nn.Conv2d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1)
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        # 并行处理各尺度卷积分支
        branch_outputs = [branch(x) for branch in self.branches]
        # 在通道维度上拼接多个尺度的输出
        out = torch.cat(branch_outputs, dim=1)
        # print(f"After cat out shape:{out.shape}")
        # 融合多尺度特征
        out = self.fuse(out)
        # print(f"After fuse out shape:{out.shape}")
        # 宽度对齐处理
        if out.size(3) != residual.size(3):
            min_width = min(out.size(3), residual.size(3))
            out = out[:, :, :, :min_width]
            residual = residual[:, :, :, :min_width]

        out += residual
        out = self.relu(out)
        return out

class TCN(nn.Module):
    def __init__(self, n_classes):
        """
        -需要修改的地方：
        - 保证输出的内容是[batch_size, H*W,C]
        - 而不需要average pooling
        """
        super(TCN, self).__init__()
        self.initial_conv = nn.Conv2d(4, 16, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()

        # 使用多尺度残差块替代原有单尺度块
        self.block1 = MultiScaleResidualBlock(16, 32, kernel_sizes=[3,5,7], dilation=1)
        self.block2 = MultiScaleResidualBlock(32, 64, kernel_sizes=[3,5,7], dilation=2)
        self.block3 = MultiScaleResidualBlock(64, 128, kernel_sizes=[3,5,7], dilation=4)
        self.block4 = MultiScaleResidualBlock(128, 256, kernel_sizes=[3,5,7], dilation=8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 256))
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x): 
        # print(f"the shape in TCN firt appear is {x.shape}")
        out = self.relu(self.initial_conv(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.squeeze(2)
        # print(f"ClassificationModelMultiScale before AdaptiveAvgPool2d out shape:{out.shape}")
        # ClassificationModelMultiScale before AdaptiveAvgPool2d out shape:torch.Size([2, 128, 1, 100])
        return out

if __name__ == '__main__':
    # 创建模型实例并打印模型结构
    # 初始化模型
    n_classes = 10
    model = TCN(n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 打印模型摘要信息
    summary(model, input_size=(4, 1, 100))  # 输入形状为 (channels, height, width)