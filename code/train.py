import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data
from data.preprocess import load_data_cache
from utils import next_batch
from models import PrototypicalNetwork
from config import TASK_NUM, NUM_EPOCHS, DATA_DIR
import config
from torchinfo import summary
import time
import os
import csv

def evaluate_performance(model, x_support, y_support, x_query, y_query):
    with torch.no_grad():
        model.eval()  # 设置为评估模式
        loss, accuracy = model(x_support, x_query, y_support, y_query)
        model.train()  # 回到训练模式
    return loss.item(), accuracy.item()

def test_model_on_multiple_tasks(model, device, datasets, datasets_cache, num_tasks=TASK_NUM):
    model.eval()  # 评估模式
    total_test_acc = 0
    total_test_loss = 0
    for _ in range(num_tasks):
        x_spt, y_spt, x_qry, y_qry = next_batch('test', datasets, datasets_cache)  # 获取测试数据
        x_spt = torch.from_numpy(x_spt).to(device)
        y_spt = torch.from_numpy(y_spt).long().to(device)
        x_qry = torch.from_numpy(x_qry).to(device)
        y_qry = torch.from_numpy(y_qry).long().to(device)

        for i in range(x_spt.size(0)):
            test_loss, test_acc = evaluate_performance(model, x_spt[i], y_spt[i], x_qry[i], y_qry[i])
            total_test_acc += test_acc
            total_test_loss += test_loss

    model.train()  # 切换回训练模式
    average_test_acc = total_test_acc / (num_tasks * x_spt.size(0))
    average_test_loss = total_test_loss / (num_tasks * x_spt.size(0))
    return average_test_loss, average_test_acc


def train_model():
    # 加载数据
    datasets = load_data()
    datasets_cache = {mode: load_data_cache(data) for mode, data in datasets.items()}

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalNetwork().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 打开CSV文件并写入头部
    csv_filename = config.get_csv_filename(config.SNR)
    print(f"CSV Filename: {csv_filename}")
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Test Loss', 'Test Accuracy', 'Inference Time'])

        for epoch in range(NUM_EPOCHS):
            model.train()
            batch_data = next_batch('train', datasets, datasets_cache)
            x_spts, y_spts, x_qrys, y_qrys = [torch.tensor(d).to(device) for d in batch_data]

            # 初始化累积损失和准确率
            total_loss = 0.0
            total_acc = 0.0
            total_inference_time = 0.0

            start = time.time()

            # 遍历每个任务
            for i in range(x_spts.size(0)):
                current_x_spt = x_spts[i]
                current_y_spt = y_spts[i]
                current_x_qry = x_qrys[i]
                current_y_qry = y_qrys[i]

                # 记录前向传播开始时间
                inference_start = time.time()

                # 前向传播
                # print(f"current_x_spt shape: {current_x_spt.shape}")
                loss, acc = model(current_x_spt, current_x_qry, current_y_spt, current_y_qry)
                
                # 记录前向传播结束时间
                inference_end = time.time()

                # 计算推理时间
                inference_time = inference_end - inference_start

                total_loss += loss
                total_acc += acc
                total_inference_time += inference_time

                # 可以在此输出或记录每个任务的推理时间
                # print(f"Task {i+1}/{x_spts.size(0)} - Inference Time: {inference_time:.4f}s")

            # 计算平均损失和准确率
            average_loss = total_loss / x_spts.size(0)
            average_acc = total_acc / x_spts.size(0)
            # 计算平均推理时间
            
            average_inference_time = total_inference_time / x_spts.size(0)
            # 输出平均推理时间
            print(f"Average Inference Time: {average_inference_time:.4f}s")
            
            # 反向传播和优化
            optimizer.zero_grad()
            average_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 最大梯度裁剪值为5.0
            optimizer.step()

            end = time.time()

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Avg Loss {average_loss.item():.4f}, Avg Accuracy {average_acc.item():.4f}, Time {end - start:.2f}s")

            if epoch % 10 == 0:
                test_loss, test_accuracy = test_model_on_multiple_tasks(model, device, datasets, datasets_cache)
                print(f"After Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
                csv_writer.writerow([epoch+1, test_loss, test_accuracy, inference_time])

    print("Training Completed!")

if __name__ == "__main__":
    train_model()
