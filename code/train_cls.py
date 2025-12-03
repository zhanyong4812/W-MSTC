import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import config
from models import ClassificationNetwork
from data.cls_data_loader import load_flat_data


class StandardModalDataset(Dataset):
    """
    Split a "frame stack" sample into two parts:
      - First stack_img_size frames: Constellation diagram sequence → ConvLSTM
      - Remaining frames: IQ diagram sequence → Flattened for TCN input
    Uses channels.stack_size / stack_img_size from code/config.yaml for division.
    """

    def __init__(self, stacks: np.ndarray, labels: np.ndarray):
        """
        stacks: [N, STACK, 1, 32, 32]
        labels: [N,]
        """
        self.stacks = stacks
        self.labels = labels

        assert stacks.ndim == 5 and stacks.shape[2] == 1 and stacks.shape[3] == 32 and stacks.shape[4] == 32, (
            "stacks must have shape [N, STACK, 1, 32, 32]"
        )
        expected_stack = config.STACK_SIZE
        assert stacks.shape[1] == expected_stack, (
            f"Each sample's STACK must equal config.STACK_SIZE = {expected_stack}"
        )

        self.img_stack = config.STACK_IMG_SIZE
        self.iq_stack = config.STACK_SIZE - config.STACK_IMG_SIZE

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        stack = self.stacks[idx]  # [STACK, 1, 32, 32]
        label = int(self.labels[idx])

        # Constellation diagram part: [IMG_STACK, 1, 32, 32] -> [IMG_STACK, 32, 32]
        img_seq = stack[: self.img_stack]
        img_seq = torch.from_numpy(img_seq).float().squeeze(1)

        # IQ part: [IQ_STACK, 1, 32, 32] -> [IQ_STACK, 1, 1024]
        iq_seq = stack[self.img_stack :]
        iq_seq = iq_seq.reshape(self.iq_stack, 1, 32 * 32)
        iq_seq = torch.from_numpy(iq_seq).float()

        return img_seq, iq_seq, label


def train_cls_model():
    # 1. Load flattened sample data
    data = load_flat_data(test_size=0.2, random_state=42)
    train_stacks = data["train"]["img"]
    train_labels = data["train"]["label"]
    test_stacks = data["test"]["img"]
    test_labels = data["test"]["label"]

    # 2. Construct Dataset / DataLoader
    train_dataset = StandardModalDataset(train_stacks, train_labels)
    test_dataset = StandardModalDataset(test_stacks, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CLS_BATCH_SIZE,
        shuffle=True,
        num_workers=config.CLS_NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.CLS_BATCH_SIZE,
        shuffle=False,
        num_workers=config.CLS_NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4. CSV logging
    csv_path = config.get_csv_filename(config.SNR)
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc = 0.0
    os.makedirs(config.CLS_SAVE_DIR, exist_ok=True)

    # 5. Training + Validation
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for img_seq_batch, iq_seq_batch, labels in train_loader:
            img_seq_batch = img_seq_batch.to(device)
            iq_seq_batch = iq_seq_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(img_seq_batch, iq_seq_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img_seq_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += torch.sum(preds == labels).item()
            total_samples += img_seq_batch.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for img_seq_batch, iq_seq_batch, labels in test_loader:
                img_seq_batch = img_seq_batch.to(device)
                iq_seq_batch = iq_seq_batch.to(device)
                labels = labels.to(device)

                logits = model(img_seq_batch, iq_seq_batch)
                loss = criterion(logits, labels)

                val_loss += loss.item() * img_seq_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += torch.sum(preds == labels).item()
                val_samples += img_seq_batch.size(0)

        val_loss_epoch = val_loss / val_samples
        val_acc_epoch = val_correct / val_samples

        print(
            f"[CLS][Epoch {epoch}/{config.NUM_EPOCHS}] "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}  |  "
            f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}"
        )

        # Log to CSV
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    epoch,
                    f"{epoch_loss:.4f}",
                    f"{epoch_acc:.4f}",
                    f"{val_loss_epoch:.4f}",
                    f"{val_acc_epoch:.4f}",
                ]
            )

        # Record best model (currently only records best accuracy, can save weights if needed)
        if val_acc_epoch > best_acc:
            best_acc = val_acc_epoch

    print(f"[CLS] Training finished! Best Val Acc: {best_acc:.4f}")
    print(f"[CLS] Training log saved to: {csv_path}")


if __name__ == "__main__":
    train_cls_model()


