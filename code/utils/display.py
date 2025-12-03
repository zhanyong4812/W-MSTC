import config


def display_config():
    """Print the key runtime configuration for the current experiment."""
    print("======= Current Configuration =======")
    print("[Data]")
    print(f"SNR: {config.SNR}")
    print(f"DATASET: {config.TRAIN_DATA}")
    print(f"RESULT_DIR: {config.DATA_DIR}")

    print("\n[Task]")
    print(f"N_WAY: {config.N_WAY}, K_SPT: {config.K_SPT}, K_QUERY: {config.K_QUERY}")
    print(f"TASK_NUM=BATCH_SIZE: {config.TASK_NUM}, NUM_EPOCHS: {config.NUM_EPOCHS}")

    print("\n[Model]")
    print(f"IMG_DIM: {config.IMG_DIM}, IQAP_DIM: {config.IQAP_DIM}")
    print(f"EMBED_DIM: {config.EMBED_DIM}, NUM_HEADS: {config.NUM_HEADS}")
    print(f"SEQ_LENGTH: {config.SEQ_LENGTH}, WINDOW_SIZES: {config.WINDOW_SIZES}")
    print(f"SWIN_LAYERS: {config.swin_params['num_layers']}, DROP: {config.swin_params['drop']}")

    print("\n[Fusion Modules]")
    print(f"WA: {config.branch['use_wa']}, MWA: {config.branch['use_mwa']}, CA: {config.branch['use_ca']}")

    csv_filename = config.get_csv_filename(config.SNR)
    print("\n[Logging]")
    print(f"CSV_FILENAME: {csv_filename}")
    print("=====================================")
