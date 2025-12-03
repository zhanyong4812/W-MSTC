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
    """
    Evaluate model performance on one support/query split.
    """
    with torch.no_grad():
        model.eval()  # Set to evaluation mode
        loss, accuracy = model(x_support, x_query, y_support, y_query)
        model.train()  # Switch back to training mode
    return loss.item(), accuracy.item()

def test_model_on_multiple_tasks(model, device, datasets, datasets_cache, num_tasks=TASK_NUM):
    """
    Test the model on multiple tasks and return average loss and accuracy.
    """
    model.eval()  # Evaluation mode
    total_test_acc = 0
    total_test_loss = 0
    for _ in range(num_tasks):
        x_spt, y_spt, x_qry, y_qry = next_batch('test', datasets, datasets_cache)  # Get test data
        x_spt = torch.from_numpy(x_spt).to(device)
        y_spt = torch.from_numpy(y_spt).long().to(device)
        x_qry = torch.from_numpy(x_qry).to(device)
        y_qry = torch.from_numpy(y_qry).long().to(device)

        for i in range(x_spt.size(0)):
            test_loss, test_acc = evaluate_performance(model, x_spt[i], y_spt[i], x_qry[i], y_qry[i])
            total_test_acc += test_acc
            total_test_loss += test_loss

    model.train()  # Switch back to training mode
    average_test_acc = total_test_acc / (num_tasks * x_spt.size(0))
    average_test_loss = total_test_loss / (num_tasks * x_spt.size(0))
    return average_test_loss, average_test_acc


def train_model():
    # Load data
    datasets = load_data()
    datasets_cache = {mode: load_data_cache(data) for mode, data in datasets.items()}

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalNetwork().to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Open CSV file and write header
    csv_filename = config.get_csv_filename(config.SNR)
    print(f"CSV Filename: {csv_filename}")
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Test Loss', 'Test Accuracy', 'Inference Time'])

        for epoch in range(NUM_EPOCHS):
            model.train()
            batch_data = next_batch('train', datasets, datasets_cache)
            x_spts, y_spts, x_qrys, y_qrys = [torch.tensor(d).to(device) for d in batch_data]

            # Accumulate metrics for each task to avoid multi-task graph mutual references
            loss_list = []
            acc_values = []
            inference_times = []

            start = time.time()

            # Iterate over each task
            for i in range(x_spts.size(0)):
                current_x_spt = x_spts[i]
                current_y_spt = y_spts[i]
                current_x_qry = x_qrys[i]
                current_y_qry = y_qrys[i]

                # Record inference start time
                inference_start = time.time()

                # Forward pass
                loss, acc = model(current_x_spt, current_x_qry, current_y_spt, current_y_qry)

                # Record inference end time
                inference_end = time.time()

                # Calculate inference time
                inference_time = inference_end - inference_start

                loss_list.append(loss)
                acc_values.append(acc.item())
                inference_times.append(inference_time)

                # You can output or log each task's inference time here
                # print(f"Task {i+1}/{x_spts.size(0)} - Inference Time: {inference_time:.4f}s")

            # Calculate average loss and accuracy
            average_loss = torch.stack(loss_list).mean()
            average_acc = sum(acc_values) / len(acc_values)

            # Calculate average inference time
            average_inference_time = sum(inference_times) / len(inference_times)

            # Print average inference time
            print(f"Average Inference Time: {average_inference_time:.4f}s")
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            average_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Clip gradients to max norm of 5.0
            optimizer.step()

            end = time.time()

            if epoch % 1 == 0:
                # average_loss is a tensor, average_acc is a Python float
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Avg Loss {average_loss.item():.4f}, Avg Accuracy {average_acc:.4f}, Time {end - start:.2f}s")

            if epoch % 10 == 0:
                test_loss, test_accuracy = test_model_on_multiple_tasks(model, device, datasets, datasets_cache)
                print(f"After Epoch {epoch+1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
                csv_writer.writerow([epoch+1, test_loss, test_accuracy, average_inference_time])

    print("Training Completed!")

if __name__ == "__main__":
    train_model()
