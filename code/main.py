import argparse
import config  # Import configuration module
from models import summary_mode
from train import train_model
import subprocess
import os
from utils import display_config

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Run Prototypical Network with different modes and support set sample counts."
    )
    
    # Add mode argument
    parser.add_argument('--mode', type=str, choices=['main', 'sum'], default='main', 
                        help="Choose the mode to run: 'main' for training, 'sum' for summary display.")
    # Add support set sample count argument
    parser.add_argument('--k_spt', type=int, default=None,
                        help="Override the default support set sample number (K_SPT) from config.py.")
    # Add N_WAY argument
    parser.add_argument('--n_way', type=int, default=None,
                        help="Override the default N_WAY value from config.py.")
    # Add SNR argument
    parser.add_argument('--snr', type=int, default=6,
                        help="Specify the SNR value to select the dataset (e.g., 6, -16, 8).")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Override K_SPT if provided
    if args.k_spt is not None:
        config.K_SPT = args.k_spt
        print(f"Using custom K_SPT = {config.K_SPT}")
    
    # Override N_WAY if provided
    if args.n_way is not None:
        config.N_WAY = args.n_way
        print(f"Using custom N_WAY = {config.N_WAY}")
    
    # Set SNR and update dataset path
    config.SNR = args.snr
    print(f"Using dataset for SNR = {config.SNR}")
    config.TRAIN_DATA = config.get_dataset_path(config.SNR)
    config.DATA_DIR = os.path.join(config.DATA_DIR, f"SNR_{config.SNR}")
    os.makedirs(config.DATA_DIR, exist_ok=True)  # Create output directory for this SNR
    
    # Display current configuration
    display_config()
    
    # Execute based on mode
    if args.mode == 'sum':
        summary_mode()
    else:
        train_model()

if __name__ == "__main__":
    main()
