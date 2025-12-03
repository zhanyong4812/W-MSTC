# split_radioML2018_tools

This repository provides a complete pipeline for:

1. Extracting IQ data from an HDF5 dataset
2. Stacking IQ samples across modulation types
3. Generating constellation diagrams from stacked IQ
4. Combining constellation maps with raw IQ for final model input

## Repository Structure

```
├── config/
│   └── config.yaml          # YAML configuration file
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── scripts/
    ├── config_reader.py     # Read and parse config.yaml
    ├── constellation_utils.py  # Map IQ to constellation grid
    ├── data_extractor.py    # Extract IQ from HDF5
    ├── main.py              # Entry point for the pipeline
    ├── stack_utils.py       # Stack IQ and combine data
    └── __init__.py

```

## Installation

1. **Clone** this repository:

```bash
git clone https://github.com/zhanyong4812/W-MSTC.git
cd W-MSTC/tools/
```

1. **Create** a Python virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

1. **Install** dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to set your dataset paths and parameters. Example:

```yaml
hdf5_path: "/data/dataset/MLRadio/RML2018.01a/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
output_dir: "/data/dataset/MLRadio/RadioML2018/"

modulations: all
snrs: [6]
samples: 1000
snr: [6]

stack_iq_data: true
generate_constellation: true
stack_constellation_iq: true

num_timesteps_values: [8]
grid_size_values: [32]
```

## Usage

Run the pipeline from the root directory:

```bash
cd scripts/
python main.py
```

After execution, the output directory will look like:

```bash
$ tree /data/dataset/MLRadio/RadioML2018
.
├── Constellation_SNR_6_k1000_grid_size32_num_timesteps8.npy
├── IQ_SNR_6_k1000_grid_size32_num_timesteps8.npy
├── Mutil_SNR_6_k1000_size32_step8.npy
└── SNR_6/
    ├── X_SNR_6_mod_128APSK.npy
    ├── X_SNR_6_mod_128QAM.npy
    └── ... (one .npy per modulation)
```