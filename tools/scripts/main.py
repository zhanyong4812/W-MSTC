import os
import logging
import numpy as np

from config_reader import read_config
from data_extractor import extract_iq_data_from_hdf5, MODULATION_LIST, SNR_LIST
from constellation_utils import generate_constellation_data
from stack_utils import (
    stack_iq_data_from_folder,
    stack_constellation_and_iq
)

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 1) Read YAML configuration
    cfg = read_config("config.yaml")

    hdf5_path  = cfg.get('hdf5_path', None)
    output_dir = cfg.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)

    # Determine modulation types and SNR list
    if cfg.get('modulations') == "all":
        target_modulations = MODULATION_LIST
    else:
        target_modulations = cfg.get('modulations', MODULATION_LIST)

    if cfg.get('snrs') == "all":
        target_snrs = SNR_LIST
    else:
        target_snrs = cfg.get('snrs', [])

    # Sample counts
    samples_per_condition = cfg.get('samples', 1000)
    single_snr           = cfg.get('snr', 6)
    single_sample        = cfg.get('samples', 1000)

    # Read grid sizes and time steps from config
    num_timesteps_values = cfg.get('num_timesteps_values', [4])
    grid_size_values     = cfg.get('grid_size_values', [64])

    # Boolean switches for pipeline stages
    stack_iq_flag        = cfg.get('stack_iq_data', False)
    generate_const_flag  = cfg.get('generate_constellation', False)
    stack_const_iq_flag  = cfg.get('stack_constellation_iq', False)

    # Iterate over all combinations of time steps and grid sizes
    for num_timesteps in num_timesteps_values:
        for grid_size in grid_size_values:
            logging.info(f"[CONFIG] modulations={target_modulations}")
            logging.info(f"[CONFIG] snrs={target_snrs}")
            logging.info(f"[CONFIG] samples_for_extraction={samples_per_condition}")
            logging.info(f"[CONFIG] single_snr_for_BCD={single_snr}, single_samples_for_BCD={single_sample}")
            logging.info(f"[CONFIG] stack_iq_data={stack_iq_flag}, "
                         f"generate_constellation={generate_const_flag}, "
                         f"stack_constellation_iq={stack_const_iq_flag}")

            # ------------------------------------
            # A) Extract IQ data from HDF5 if configured
            # ------------------------------------
            if hdf5_path and os.path.exists(hdf5_path):
                logging.info("Starting IQ extraction from HDF5...")
                extract_iq_data_from_hdf5(
                    hdf5_path=hdf5_path,
                    output_dir=output_dir,
                    target_modulations=target_modulations,
                    target_snrs=target_snrs,
                    samples_per_condition=samples_per_condition
                )
            else:
                logging.warning("No valid hdf5_path configured or file not found; skipping IQ extraction")

            # ====================
            # B) Stack IQ data
            # ====================
            if stack_iq_flag:
                for s in target_snrs:
                    snr_folder = os.path.join(output_dir, f"SNR_{s}")
                    if os.path.exists(snr_folder):
                        combined_data, snr_val = stack_iq_data_from_folder(snr_folder)
                        out_file = os.path.join(
                            output_dir,
                            f"IQ_SNR_{snr_val}_k{single_sample}_"
                            f"grid_size{grid_size}_num_timesteps{num_timesteps}.npy"
                        )
                        np.save(out_file, combined_data)
                        logging.info(f"[B] Completed stacking for SNR={snr_val}: "
                                     f"shape={combined_data.shape}, output={out_file}")
                    else:
                        logging.warning(f"[B] Folder {snr_folder} does not exist; skipping IQ stacking")

            # ====================
            # C) Generate constellation diagrams
            # ====================
            if generate_const_flag:
                for s in target_snrs:
                    iq_file = os.path.join(
                        output_dir,
                        f"IQ_SNR_{s}_k{single_sample}_"
                        f"grid_size{grid_size}_num_timesteps{num_timesteps}.npy"
                    )
                    if os.path.exists(iq_file):
                        iq_data = np.load(iq_file)
                        constellation_out = os.path.join(
                            output_dir,
                            f"Constellation_SNR_{s}_k{single_sample}_"
                            f"grid_size{grid_size}_num_timesteps{num_timesteps}.npy"
                        )
                        generate_constellation_data(iq_data, constellation_out,
                                                    grid_size, num_timesteps)
                        logging.info(f"[C] Constellation generated: {constellation_out}")
                    else:
                        logging.warning(f"[C] File not found {iq_file}; cannot generate constellation")

            # ==========================================
            # D) Stack constellation + IQ for each SNR
            #     Output filename must match train_data_template in training-side config.yaml:
            #     Mutil_SNR_{snr}_k{samples}_size{size}_step{step}.npy
            # ==========================================
            if stack_const_iq_flag:
                for s in target_snrs:
                    const_npy = os.path.join(
                        output_dir,
                        f"Constellation_SNR_{s}_k{single_sample}_"
                        f"grid_size{grid_size}_num_timesteps{num_timesteps}.npy"
                    )
                    iq_npy = os.path.join(
                        output_dir,
                        f"IQ_SNR_{s}_k{single_sample}_"
                        f"grid_size{grid_size}_num_timesteps{num_timesteps}.npy"
                    )
                    combined_out = os.path.join(
                        output_dir,
                        f"Mutil_SNR_{s}_k{single_sample}_size{grid_size}_step{num_timesteps}.npy"
                    )
                    if os.path.exists(const_npy) and os.path.exists(iq_npy):
                        stack_constellation_and_iq(const_npy, iq_npy,
                                                  combined_out,
                                                  grid_size, num_timesteps)
                        logging.info(f"[D] Constellation + IQ stacked, output: {combined_out}")
                    else:
                        logging.warning(f"[D] Missing {const_npy} or {iq_npy}; cannot stack")

if __name__ == "__main__":
    main()
