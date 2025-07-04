import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import config
from models import ClassificationNetwork
from data.data_loader import load_data  # 返回 {'train':{'img','label'}, 'test':{'img','label'}}


class StandardModalDataset(Dataset):
    """
    把一个“帧堆栈”样本拆成两部分：
      - 前 config.IMG_STACK 帧：星座图序列 → ConvLSTM
      - 后 config.IQ_STACK 帧：IQ 图序列 → 扁平成 TCN 输入
    """
    def __init__(self, stacks: np.ndarray, labels: np.ndarray):
        """
        stacks: numpy array, 形状 [N, STACK, 1, 32, 32]
        labels: numpy array, 形状 [N,]
        """
        self.stacks = stacks
        self.labels = labels

        # 验证数据维度
        assert stacks.ndim == 5 and stacks.shape[2] == 1 \
               and stacks.shape[3] == 32 and stacks.shape[4] == 32, \
               "stacks 维度必须是 [N, STACK, 1, 32, 32]"
        assert stacks.shape[1] == config.IMG_STACK + config.IQ_STACK, \
               f"每个样本的 STACK 必须等于 IMG_STACK+IQ_STACK = {config.IMG_STACK + config.IQ_STACK}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. 取出样本的整个帧堆栈： shape = [STACK, 1, 32, 32]
        stack = self.stacks[idx]
        label = int(self.labels[idx])

        # 2. 切分前后两部分
        # 2.1 星座图部分：前 config.IMG_STACK 帧 → [IMG_STACK, 1, 32, 32]
        img_seq = stack[: config.IMG_STACK]            # numpy, shape = (IMG_STACK, 1, 32, 32)
        # 转为 Tensor 并去掉 channel=1 维度 → [IMG_STACK, 32, 32]
        img_seq = torch.from_numpy(img_seq).float().squeeze(1)

        # 2.2 IQ 部分：后 config.IQ_STACK 帧 → [IQ_STACK, 1, 32, 32]
        iq_seq = stack[config.IMG_STACK : ]            # numpy, shape = (IQ_STACK, 1, 32, 32)
        # 扁平化到 (IQ_STACK, 1, 1024)
        iq_seq = iq_seq.reshape(config.IQ_STACK, 1, 32 * 32)
        iq_seq = torch.from_numpy(iq_seq).float()      # Tensor, shape = [IQ_STACK, 1, 1024]

        # 3. 返回 (img_seq, iq_seq, label)
        # 在 DataLoader 中，会自动把 B 维拼起来：
        #   -> img_seq_batch: [B, IMG_STACK, 32, 32]
        #   -> iq_seq_batch:  [B, IQ_STACK, 1, 1024]
        return img_seq, iq_seq, label


def train_model():
    # —— 1. 加载并划分数据 —— 
    data = load_data(test_size=0.2, random_state=42)
    train_stacks = data['train']['img']    # numpy, shape (N_train, STACK, 1, 32, 32)
    train_labels = data['train']['label']  # numpy, shape (N_train,)
    test_stacks  = data['test']['img']     # numpy, shape (N_test, STACK, 1, 32, 32)
    test_labels  = data['test']['label']   # numpy, shape (N_test,)

    # —— 2. 构造 Dataset 与 DataLoader —— 
    train_dataset = StandardModalDataset(train_stacks, train_labels)
    test_dataset  = StandardModalDataset(test_stacks, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size   = config.BATCH_SIZE,
        shuffle      = True,
        num_workers  = config.NUM_WORKERS,
        pin_memory   = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size   = config.BATCH_SIZE,
        shuffle      = False,
        num_workers  = config.NUM_WORKERS,
        pin_memory   = True
    )

    # —— 3. 初始化模型、损失函数与优化器 —— 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # —— 4. 准备 CSV 日志文件 —— 
    csv_path = config.get_csv_filename(config.SNR)
    # 如果文件不存在，创建并写入表头；如果已存在则覆盖
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])

    best_acc = 0.0
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # —— 5. 训练 + 验证 循环 —— 
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for img_seq_batch, iq_seq_batch, labels in train_loader:
            # img_seq_batch: [B, IMG_STACK, 32, 32]
            # iq_seq_batch:  [B, IQ_STACK, 1, 1024]
            # labels:        [B]
            img_seq_batch = img_seq_batch.to(device)
            iq_seq_batch  = iq_seq_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(img_seq_batch, iq_seq_batch)  # [B, NUM_CLASSES]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img_seq_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += torch.sum(preds == labels).item()
            total_samples += img_seq_batch.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc  = running_correct / total_samples

        # —— 验证阶段 —— 
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for img_seq_batch, iq_seq_batch, labels in test_loader:
                img_seq_batch = img_seq_batch.to(device)
                iq_seq_batch  = iq_seq_batch.to(device)
                labels = labels.to(device)
                logits = model(img_seq_batch, iq_seq_batch)
                loss = criterion(logits, labels)

                val_loss += loss.item() * img_seq_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += torch.sum(preds == labels).item()
                val_samples += img_seq_batch.size(0)

        val_loss_epoch = val_loss / val_samples
        val_acc_epoch  = val_correct / val_samples

        # 打印当前 epoch 信息
        print(
            f"[Epoch {epoch}/{config.NUM_EPOCHS}] "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}  |  "
            f"Val   Loss: {val_loss_epoch:.4f}, Val   Acc: {val_acc_epoch:.4f}"
        )

        # —— 6. 将本 epoch 结果写入 CSV —— 
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                epoch,
                f"{epoch_loss:.4f}",
                f"{epoch_acc:.4f}",
                f"{val_loss_epoch:.4f}",
                f"{val_acc_epoch:.4f}"
            ])

        # 保存最优模型
        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch
            # torch.save(
            #     model.state_dict(),
            #     os.path.join(config.SAVE_DIR, "best_model.pth")
            # )

    print(f"Training finished! Best Val Acc: {best_acc:.4f}")
    print(f"Training log saved to: {csv_path}")


if __name__ == "__main__":
    train_model()
