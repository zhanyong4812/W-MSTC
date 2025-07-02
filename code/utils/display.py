import config


    
def display_config():
    # 打印当前运行的一些参数设置
    print("------- Current Configuration -------")
    print("------- Basic Configuration -------")
    print(f"N_WAY: {config.N_WAY}, K_SPT: {config.K_SPT}, K_QUERY: {config.K_QUERY}, TASK_NUM = BATCH_SIZE: {config.TASK_NUM}")
    print(f"NUM_EPOCHS: {config.NUM_EPOCHS}")
    # 此时如果需要使用 CSV 文件名，调用 get_csv_filename() 即可获得最新结果
    csv_filename = config.get_csv_filename(config.SNR)
    print(f"CSV_FILENAME: {csv_filename}")
#     print(f"SEQ_LENGTH: {config.SEQ_LENGTH}, EMBED_DIM: {config.EMBED_DIM}, NUM_HEADS: {config.NUM_HEADS}")
#     print(f"WINDOW_SIZES: {config.WINDOW_SIZES}")
    
#     # 统计 branch 配置中为 True 的数量
#     true_count = sum(value is True for value in config.branch.values())
#     print(f"Opened branch : {true_count}/3")
    
#     # 动态展示 branch 配置
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
