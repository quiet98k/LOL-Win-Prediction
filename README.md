# LOL-Win-Prediction

## Description

This project predics the win probability in League of Legends matches using a deep neural network trained on in-game data (first 15 minutes). It also improves the model's **local robustness** through techniques such as PGD attacks, noise injection, Jacobian regularization, and TRADES.

## Project Architecture

```bash
.
├── LICENSE
├── README.md
├── data/                           # All data-related files
│   ├── OraclesElixir/              # Raw data source from Oracles Elixir
│   ├── champion_label_mapping.csv
│   ├── minmax_scaler.pkl
│   ├── normalize_cols.json
│   ├── processed_lol_data.parquet  # Final processed dataset used for training/evaluation
│   └── skewed_features.json
├── lolnet.py                       # LoLNet model definition
├── models/                         # Directory containing all trained models
│   ├── lolnet*.pth                 # PyTorch model checkpoints
│   └── lolnet*.onnx                # Exported ONNX models for verification
├── prediction.ipynb                # Main file for data processing, model training and model evaluation
├── pyproject.toml
├── requirements.txt
├── trades.py                       # TRADES loss function for adversarial training
└── uv.lock

```

## Setup

### Option 1: Using `conda`

```bash
# Create virtual environment
conda create -n lol-robust-prediction python=3.10 -y
conda activate lol-robust-prediction

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using `venv`

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using `uv`

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# install dependencies
uv pip install -r requirements.txt
```

## Usage

### Data Processing

The file `data/processed_lol_data.parquet` is the processed dataset used for training and evaluation. If you want to modify or regenerate this data:

1. **Download Original Data**  
   Get the raw match data:  
   [Download link (Google Drive)](https://drive.google.com/drive/folders/10oDOO6hE4CTj3n-CYnfsVg1WdM_HDr06?usp=sharing)

2. **Place Files**  
   Put the downloaded files into the following directory:

   ```
   data/OraclesElixir/
   ```

3. **Run Preprocessing**
   Open `prediction.ipynb` and execute the **Exploratory Data Analysis & Preprocessing** section.
   It will Save the output as `processed_lol_data.parquet`

### Model Training

The model architecture is defined in `lolnet.py`.

To train the model, use the `train_lolnet_model()` function defined in the **Model Training** section of `prediction.ipynb`:

```python
def train_lolnet_model(df,
                       model_name,
                       is_numerical,
                       pgd=False,
                       noise=False,
                       trades=False,
                       jacobian_reg=False):
```

**Parameters:**

- `df`: The processed DataFrame (loaded from `processed_lol_data.parquet`)
- `model_name`: Name used for saving the model files
- `is_numerical`: List of booleans indicating which features are numerical and can be perturbed
- Flags like `pgd`, `noise`, `trades`, and `jacobian_reg` enable specific robustness training techniques

**Output:**

- Saves a trained model in both PyTorch and ONNX formats:
  - `models/{model_name}.pth`
  - `models/{model_name}.onnx`

**Example usage**:

```python
train_lolnet_model(df, "lolnet_trades", is_numerical, trades=True)
```

### Model Evaluation

To evaluate all trained models, execute the **Model Evaluation** section in `prediction.ipynb`. This will automatically:

- Load each model in the `models/` directory
- Run all robustness and performance tests
- Print metrics such as clean accuracy, robustness accuracy, verified robustness, gradient norms, and Lipschitz constant

---

If you'd like to run evaluations manually, you can call the following functions:

```python
def test_lolnet_clean_accuracy(
    model,
    df,
    batch_size=64,
    device='cpu'
)
```

> Evaluates model accuracy on unperturbed (clean) test data.

```python
def test_lolnet_robustness_accuracy(
    model,
    df,
    is_numerical,
    epsilon_val=0.01,
    batch_size=64,
    alpha=0.005,
    steps=20,
    device='cpu'
)
```

> Measures robustness accuracy using a PGD adversarial attack.

```python
def test_lolnet_verified_robustness_rate(
    df: pd.DataFrame,
    onnx_path: str,
    epsilon: float,
    is_numerical: list[bool],
    num_samples: int = 100,
    verbosity: int = 1
)
```

> Uses Marabou for formal verification of local robustness.

```python
def get_gradient_norms(
    model,
    df: pd.DataFrame,
    is_numerical: list[bool],
    norm_type=2,
    batch_size=64
)
```

> Computes gradient norms for the model.

```python
def compute_lipschitz_constant(model)
```

> Estimates the model’s Lipschitz constant
