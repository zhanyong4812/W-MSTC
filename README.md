# W-MSTC



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
   cd W-MSTC/
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

```less
code
├── config.py / config.yaml
├── main.py / train.py / train_cls.py
├── models/
│   ├── classification_network.py
│   ├── prototypical_network.py
│   ├── feature_extraction/
│   │   ├── Conv_LSTM.py
│   │   └── MultiScale_TCN.py
│   └── feature_fusion/
│       ├── window_attention.py
│       ├── multi_window_attention.py
│       ├── cross_attention.py
│       └── windowed_cross_attention.py
├── data/
│   ├── data_loader.py
│   ├── cls_data_loader.py
│   └── preprocess.py
├── utils/
│   ├── config_utils.py
│   ├── display.py
│   └── helpers.py
├── results/
│   └── SNR_x/ 
└── requirements.txt
```

------

### Data Preparation

1. Obtain the RadioML 2018 dataset.
2. Use the provided `tools` to generate `.npy` files.
3. See the `tools/README.md` for detailed instructions.
4. Update paths in `config.py` to point to your `.npy` directory.

```bash
cd tools/scripts
python main.py
```

------

### Usage

Run **main** with your desired few-shot settings (e.g., 5-way 5-shot):

```bash
cd code
python main.py --k_spt 5 --n_way 5
```

Key flags in `main.py`:

- `--k_spt` : number of support examples per class
- `--n_way` : number of classes per episode

------

### Output

After running experiments, results are saved under `results/{SNR}` with timestamped CSV files:

```less
results
└── SNR_6
```

### TODO

- **Decouple** data-loading, model, and training logic for easier reuse.
- **Publish visualization + log tooling** in dev branch and merge back once stabilized.
- **Add unit tests** for feature extraction and fusion modules.

---

 More visualization work and subsequent experimentation/updates will continue on the `dev` branch; `main` only tracks stable releases. 

### Contact

This project is maintained by **zhanyong4812**, reachable at **[zhanyong4812](2024220603114@mails.zstu.edu.cn)** for questions, requests, or contributions.



