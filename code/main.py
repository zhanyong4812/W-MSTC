# main.py
import argparse
import os
import config         # 引入全局配置
from train import train_model
from utils import display_config

def main():
    # 1. 创建 ArgumentParser，声明支持的命令行参数
    parser = argparse.ArgumentParser(
        description="Run Prototypical Network with different modes and support set sample counts."
    )
    parser.add_argument(
        '--mode', type=str, choices=['main', 'sum'], default='main',
        help="Choose the mode to run: 'main' for training, 'sum' for summary display."
    )
    parser.add_argument(
        '--k_spt', type=int, default=None,
        help="Override the default support set sample number (K_SPT) from config.py."
    )
    parser.add_argument(
        '--n_way', type=int, default=None,
        help="Override the default N_WAY value from config.py."
    )
    parser.add_argument(
        '--snr', type=int, default=6,
        help="Specify the snr value to select the dataset (e.g., 6, -16, 8)."
    )

    # 2. 解析命令行
    args = parser.parse_args()

    # 3. 如果用户通过 --k_spt 覆盖默认配置，就在 config 中更新
    # if args.k_spt is not None:
    #     config.K_SPT = args.k_spt
    #     print(f"Using custom K_SPT = {config.K_SPT}")
    # # 4. 如果用户通过 --n_way 覆盖默认配置，就在 config 中更新
    # if args.n_way is not None:
    #     config.N_WAY = args.n_way
    #     print(f"Using custom N_WAY = {config.N_WAY}")
    # 5. 设置 snr、以及对应的数据路径
    config.SNR = args.snr
    print(f"Using dataset for snr = {config.SNR}")
    config.TRAIN_DATA = config.get_dataset_path(config.SNR)
    config.DATA_DIR = os.path.join(config.DATA_DIR, f"SNR_{config.SNR}")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # 6. 打印当前配置（便于调试）
    display_config()

    # 7. 根据 --mode 决定是调用 summary_mode 还是 train_model
    if args.mode == 'sum':
        print("TODO summary_mode()")
    else:
        train_model()

if __name__ == "__main__":
    main()
