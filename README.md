# W-MSTC



## Split Tools

This repository provides a complete pipeline for:

1. Extracting IQ data from an HDF5 dataset
2. Stacking IQ samples across modulation types
3. Generating constellation diagrams from stacked IQ
4. Combining constellation maps with raw IQ for final model input

### Repository Structure

```
├── config/
│   └── config.yaml          # YAML configuration file
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── scripts/
│   ├── config_reader.py     # Read and parse config.yaml
│   ├── constellation_utils.py  # Map IQ to constellation grid
│   ├── data_extractor.py    # Extract IQ from HDF5
│   ├── main.py              # Entry point for the pipeline
│   ├── stack_utils.py       # Stack IQ and combine data
│   └── __init__.py
└── tests/
    └── check_npy.ipynb      # Notebook for verifying outputs
```

### Installation

1. **Clone** this repository:

```bash
git clone https://github.com/zhanyong4812/W-MSTC.git
cd W-MSTC/data/
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

### Configuration

Edit `config/config.yaml` to set your dataset paths and parameters. 


### Usage

Run the pipeline from the root directory:

```bash
cd scripts/
python main.py
```

After execution, the output directory will look like:

```bash
├── Constellation_SNR_6_k1000_grid_size32_num_timesteps8.npy
├── IQ_SNR_6_k1000_grid_size32_num_timesteps8.npy
├── Mutil_SNR_6_k1000_grid_size32_num_timesteps8.npy
└── SNR_6/
    ├── X_SNR_6_mod_128APSK.npy
    ├── X_SNR_6_mod_128QAM.npy
    └── ... (one .npy per modulation)
```





## Method



### Requirements

- **Python** ≥ 3.8  
- **CUDA Toolkit** 11.3  
- **PyTorch** 1.11.0  
- Other dependencies listed in `requirements.txt`

---

### Installation

1. Clone the repository:  

   ```bash
   git clone https://github.com/zhanyong4812/W-MSTC.git
   cd W-MSTC/code/
   ```

1. (Optional) Create & activate a virtual environment:

   ```bash
   python3.8 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

------

### Project Structure

```
├── config.py              
├── main.py                 
├── train.py                
├── run_experiments.py      
├── data  
├── models
├── utils     
└── results                
```

------

### Data Preparation

1. Obtain the RadioML 2018 dataset.
2. Use the provided `split_radioML2018_tools` to generate `.npy` files.
3. See the `data/README.md` for detailed instructions.
4. Update paths in `config.py` to point to your `.npy` directory.

------

### Usage

Run **main** with your desired few-shot settings (e.g., 5-way 5-shot):

```bash
python main.py --k_spt 5 --n_way 5
```

Key flags in `main.py`:

- `--k_spt` : number of support examples per class
- `--n_way` : number of classes per episode

------



### Output

After running experiments, results are saved under `results/{SNR}` with timestamped CSV files:

```
results
└── SNR_6
```



### TODO

- **Decouple** data-loading, model, and training logic for easier reuse.

- **Add hyperparameter sweeps** via YAML/JSON configs.

  



