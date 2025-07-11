import config

def display_config():
    # Print the current runtime parameter settings
    print("------- Current Configuration -------")
    print("------- Basic Configuration -------")
    print(f"N_WAY: {config.N_WAY}, K_SPT: {config.K_SPT}, K_QUERY: {config.K_QUERY}, TASK_NUM = BATCH_SIZE: {config.TASK_NUM}")
    print(f"NUM_EPOCHS: {config.NUM_EPOCHS}")
    # If you need the CSV filename at this point, call get_csv_filename() to obtain the latest result
    csv_filename = config.get_csv_filename(config.SNR)
    print(f"CSV_FILENAME: {csv_filename}")
#     print(f"SEQ_LENGTH: {config.SEQ_LENGTH}, EMBED_DIM: {config.EMBED_DIM}, NUM_HEADS: {config.NUM_HEADS}")
#     print(f"WINDOW_SIZES: {config.WINDOW_SIZES}")
    
#     # Count how many branches in the configuration are enabled
#     true_count = sum(value is True for value in config.branch.values())
#     print(f"Opened branch : {true_count}/3")
    
#     # Dynamically display branch configuration
#     if config.branch["use_branch1"]:
#         print("------- Branch1 Configuration -------")
#         print("Using Swin-Transformer.")
#     if config.branch["use_branch2"]:
#         print("------- Branch2 Configuration -------")
#         print("Using Multi-scale Self-attention.")
#     if config.branch["use_branch3"]:
#         print("------- Branch3 Configuration -------")
#         print("Using Cross-attention with Swin-Transformer.")
    
#     print("--------------------------------------")
