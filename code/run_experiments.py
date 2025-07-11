import subprocess

# Define the SNR values to process
snr_values = [30]
# snr_values = list(range(2, 31, 2))
# snr_values = list(range(-20, 32, 2))

# Define the ranges for K_SPT and N_WAY to test
k_values = [3, 5, 7, 9]
# k_values = [5, 7, 9]
# n_values = [2, 3, 5, 7]
# n_values = [2, 3, 7]

# For each SNR value, test different combinations of K_SPT and N_WAY
for snr in snr_values:
    # Test for each K_SPT
    for k in k_values:
        print("=" * 40)
        print(f"Running main.py with SNR = {snr}, K_SPT = {k}")
        print("=" * 40)
        ret = subprocess.call(["python", "main.py", "--snr", str(snr), "--k_spt", str(k)])
        if ret != 0:
            print(f"Error occurred for SNR = {snr}, K_SPT = {k}. Exiting.")
            break

    # Test for each N_WAY
    # for n in n_values:
    #     print("=" * 40)
    #     print(f"Running main.py with SNR = {snr}, N_WAY = {n}")
    #     print("=" * 40)
    #     ret = subprocess.call(["python", "main.py", "--snr", str(snr), "--n_way", str(n)])
    #     if ret != 0:
    #         print(f"Error occurred for SNR = {snr}, N_WAY = {n}. Exiting.")
    #         break
