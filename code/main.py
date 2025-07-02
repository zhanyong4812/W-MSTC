import argparse
import config  # 引入配置模块
from models import summary_mode
from train import train_model
import subprocess
import os
from utils import display_config

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Run Prototypical Network with different modes and support set sample counts."
    )
    
    # 添加运行模式参数
    parser.add_argument('--mode', type=str, choices=['main', 'sum'], default='main', 
                        help="Choose the mode to run: 'main' for training, 'sum' for summary display.")
    # 添加支持集样本数量参数
    parser.add_argument('--k_spt', type=int, default=None,
                        help="Override the default support set sample number (K_SPT) from config.py.")
    # 添加 N_WAY 参数
    parser.add_argument('--n_way', type=int, default=None,
                        help="Override the default N_WAY value from config.py.")
    # 添加 SNR 参数
    parser.add_argument('--snr', type=int, default=6,
                        help="Specify the SNR value to select the dataset (e.g., 6, -16, 8).")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果传入了 k_spt 参数，则覆盖配置文件中的默认值
    if args.k_spt is not None:
        config.K_SPT = args.k_spt
        print(f"Using custom K_SPT = {config.K_SPT}")
    
    # 如果传入了 n_way 参数，则覆盖配置文件中的默认值
    if args.n_way is not None:
        config.N_WAY = args.n_way
        print(f"Using custom N_WAY = {config.N_WAY}")
    
    # 设置SNR
    config.SNR = args.snr
    print(f"Using dataset for SNR = {config.SNR}")
    config.TRAIN_DATA = config.get_dataset_path(config.SNR)
    config.DATA_DIR = os.path.join(config.DATA_DIR, f"SNR_{config.SNR}")
    os.makedirs(config.DATA_DIR, exist_ok=True)  # Create the SNR directory if it doesn't exist
    
    # 打印当前配置
    display_config()
    
    # 根据模式执行相应操作
    if args.mode == 'sum':
        summary_mode()
    else:
        train_model()

if __name__ == "__main__":
    main()
