# Mamba-2-Project
This project investigates egocentric action prediction by combining Vision Transformers for spatial features and Mamba-2 for sequence modeling.

---

## 🚀 Features

- Use Vision Transformer to extract frame-level features  
- Multi-region Mamba-2 blocks for spatial information processing  
- Multiple iterative model variants integrated in a single codebase  
- Simple shell scripts for one-click training and validation  

---

## 📂 Project Structure

```
.
├── requirements.txt                # Environment dependencies
├── build_lmdb/                     
│   └── build_lmdb_100.py          # Script to extract ViT features and build LMDB datasets
├── rulstm2/
│   └── RULSTM/                     
│       ├── models.py               # Definitions of various model iterations
│       ├── run.sh                  # EPIC-55 default training script
│       ├── run_epic100.sh          # EPIC-100 training script
│       └── run_validation.sh       # Validation/testing script
└── README.md                       # This file
```

---

## 🛠 Environment Setup

1. **Create and activate Conda environment**  
   ```bash
   conda create -n action_pred python=3.10
   conda activate action_pred
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎬 Quick Start

Enter the core code directory to run training or validation scripts:

```bash
cd rulstm2/RULSTM

# Train on EPIC-55 (default settings)
bash run.sh

# Train on EPIC-100
bash run_epic100.sh

# Validate/Test (adjust model paths and parameters as needed)
bash run_validation.sh
```

> **Tip**: The scripts load model classes defined in `models.py` by default. Modify the script or `models.py` to switch between different model variants.

---

## 📝 Model Description

All model variants are defined in:
```
rulstm2/RULSTM/models.py
```