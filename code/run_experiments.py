import subprocess

# 定义需要处理的SNR值
snr_values = [30]
# snr_values = list(range(2, 31, 2))
# snr_values = list(range(-20, 32, 2))
# 定义需要测试的 K_SPT 和 N_WAY 范围
k_values = [3, 5, 7, 9]
# k_values = [5, 7, 9]
# n_values = [2,3,5,7]
# n_values = [2,3,7]

# 对于每个SNR值，分别测试不同的 K_SPT 和 N_WAY 组合
for snr in snr_values:
    # 针对K_SPT进行测试
    for k in k_values:
        print("=" * 40)
        print(f"Running main.py with SNR = {snr}, K_SPT = {k}")
        print("=" * 40)
        ret = subprocess.call(["python", "main.py", "--snr", str(snr), "--k_spt", str(k)])
        if ret != 0:
            print(f"Error occurred for SNR = {snr}, K_SPT = {k}. Exiting.")
            break

    # 针对N_WAY进行测试
    # for n in n_values:
    #     print("=" * 40)
    #     print(f"Running main.py with SNR = {snr}, N_WAY = {n}")
    #     print("=" * 40)
    #     ret = subprocess.call(["python", "main.py", "--snr", str(snr), "--n_way", str(n)])
    #     if ret != 0:
    #         print(f"Error occurred for SNR = {snr}, N_WAY = {n}. Exiting.")
    #         break
