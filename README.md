# Marketing Conversion Prediction

A machine-learning pipeline for predicting whether a marketing campaign record will result in a conversion. It compares XGBoost and LightGBM models using validation Average Precision, evaluates the selected model on a held-out test set, and generates SHAP explanations.

## Project structure

```text
.
├── main.py
├── requirements.txt
├── data/
│   ├── marketing_data.csv
│   ├── marketing_features.csv
│   └── marketing_selected_features.csv
├── src/
│   ├── evaluate.py
│   ├── explainer.py
│   ├── preprocessing.py
│   └── train.py
├── artifacts/             # Trained model output
├── plot/                  # SHAP plot output
└── notebook/
    └── eda.ipynb
```

## Requirements

- Python
- The packages listed in `requirements.txt`
- An NVIDIA GPU and a compatible CUDA-enabled XGBoost installation for the current XGBoost configuration (`device="cuda"`). To train without a supported GPU, change that setting to `device="cpu"` in `src/train.py`.

## Setup

From the project root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

By default, the pipeline reads `data/marketing_selected_features.csv`. It must contain these columns:

- `timestamp`, `campaign`, `cost`
- `cat1` through `cat9`
- `conversion` (the target label)

The raw and intermediate CSV files in `data/` are not required by the current `main.py` run unless you use them in your own preprocessing workflow. The input file is read relative to the project root.

## Run

Run the complete training and evaluation pipeline from the project root:

```bash
python main.py
```

The pipeline splits the data into training, validation, and test sets (70% / 15% / 15%), stratified by `conversion`. It trains XGBoost and LightGBM, selects the model with the highest validation Average Precision, and prints test-set metrics and a classification report.

## Outputs

- `artifacts/best_model.pkl` — selected trained model, saved with joblib.
- `plot/explainer.png` — SHAP beeswarm plot when the selected model is XGBoost or LightGBM.

Model and plot outputs may be ignored by Git according to `.gitignore`.
