# CMPE 257 Project: Stock Price Prediction

A machine learning project to predict stock prices using historical stock data with sliding window features and walk-forward validation.

## Project Overview

This project processes historical stock market data and prepares it for ML model training:
- **Raw Data**: Stock prices, fundamentals, and securities metadata from Kaggle
- **Processing**: Creates sliding windows (2-5 days) of features for each stock
- **Validation**: Walk-forward cross-validation (5 folds) for time-series data
- **Normalization**: StandardScaler pipeline for feature normalization
- **Goal**: Train ML models to predict next-day closing price

## Data Structure

```
data/
├── raw/                          # Kaggle CSV files (uploaded to git)
│   ├── prices.csv
│   ├── prices-split-adjusted.csv
│   ├── fundamentals.csv
│   └── securities.csv
├── naive_processed/              # Simple processing (ChatGPT baseline)
│   └── window_*.csv
├── full_processed/               # Full walk-forward validation
│   ├── X_train_window_*_fold_*.csv
│   ├── X_test_window_*_fold_*.csv
│   ├── X_eval_window_*.csv
│   ├── y_train_window_*_fold_*.csv
│   ├── y_test_window_*_fold_*.csv
│   └── y_eval_window_*.csv
└── normalized/                   # Normalized data (StandardScaler)
    └── window_*/
        ├── fold_*/
        │   ├── X_train.csv, X_test.csv, y_train.csv, y_test.csv
        │   └── scaler_pipeline.pkl
        ├── X_eval.csv
        └── y_eval.csv
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd cmpe_257_project
```

### 2. Set up Python Environment through conda or pip


### 3. Install Dependencies

Install required packages via conda:

```bash
conda install numpy pandas python-dateutil pytz six tzdata scikit-learn matplotlib -y
```

Or use pip if you prefer:

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, numpy, sklearn; print('All packages imported successfully!')"
```

**Note:** If you see `ModuleNotFoundError`, make sure conda environment is activated:
```bash
unalias python  # Remove any python aliases
conda activate cmpe257
```

## Running the Pipeline

You can either run each step manually (below) or use the automation helpers introduced in this update:

- `run_all_linear.sh`: loops through all windows 2-5 and folds 0-4 with `scripts/train_linear.py`.
- `Makefile`: provides shortcuts such as `make normalize`, `make train_linear`, `make aggregate`, `make report`, and `make full` (runs the complete chain).

Make helper scripts executable once:

```bash
chmod +x scripts/*.py run_all_linear.sh
```

### Step 1: Explore Raw Data

```bash
python scripts/initial_data_exploration.py
```

This prints dataset info (columns, dtypes, missing values) for all CSV files in `data/raw/`.

### Step 2: Process Full Data (with Walk-Forward Validation)

```bash
python scripts/process_data_full.py
```

**What it does:**
- Loads raw prices data
- Splits into train/eval by date (80/20)
- Creates sliding windows (2-5 days) for each company
- Applies walk-forward validation (5 folds)
- Generates features: open, close, low, high, volume at each time step
- Saves to `data/full_processed/`

**Duration:** ~5-10 minutes depending on machine

**Output:** Separate train/test/eval sets for each window size and fold

### Step 3: Build Normalization Pipeline

```bash
python scripts/build_pipeline.py
```

**What it does:**
- Loads processed data from `data/full_processed/`
- Creates StandardScaler for each fold
- Fits scaler on training data
- Normalizes train/test/eval sets
- Saves normalized data to `data/normalized/`
- Saves fitted scalers (`scaler_pipeline.pkl`) for later use

**Duration:** ~2-3 minutes

**Output:** Normalized datasets ready for model training

### Step 4: Explore Processed Data

```bash
python scripts/full_data_exploration.py
```

This prints statistics on the processed/normalized data and generates visualization plots.

### Step 5: Train Baseline Linear Models (all windows/folds)

```bash
./run_all_linear.sh
# or equivalently
make train_linear
```

This script calls `scripts/train_linear.py` for every window/fold combination and stores:
- Models in `models/linear_regression_*` (or ridge variants)
- Metrics JSON files in `artifacts/metrics_window_*_fold_*.json`
- Prediction plots in `reports/figs/`

### Step 6: Train Extra Models (RF / GBR / MLP)

```bash
python scripts/train_tree_nn.py --window 3 --fold 0 --model rf --params '{"n_estimators":500}'
```

Use `--model gbr` or `--model mlp` with appropriate JSON parameters to train additional regressors on any window/fold.

### Step 7: Aggregate Metrics and Build Reports

```bash
python scripts/aggregate_metrics.py     # writes reports/metrics_summary.csv
python scripts/report_plots.py          # creates reports/model_report.md + RMSE plot
```

To tune hyperparameters on any split:

```bash
python scripts/grid_search.py --window 3 --fold 0 --model ridge --param-grid '{"alpha":[0.1,1,10]}'
```

### Optional: Process Naive Data (Quick Baseline)

```bash
python scripts/process_data_naive.py
```

This creates a simple processed dataset (without walk-forward validation) for quick experiments and comparison.

---

### Config-Driven Training (`src/main.py`)

Use the orchestrator in `src/main.py` to train every model listed in a YAML config and save detailed results under `results/`.

```bash
# Baseline linear + polynomial regression
python -m src.main --config configs/baseline.yaml

# Advanced XGBoost + LSTM experiments and other models (saves fitted models)
python -m src.main --config configs/all_models.yaml.yaml
```

Config anatomy:
- `config_name`: label for the experiment folder inside `results/`.
- `windows`: sliding-window sizes to iterate over.
- `models`: collection of `{name, params}` entries. Available names now include `linear_regression`, `polynomial_regression`, `random_forest_regressor`, `gradient_boosting_regressor`, `mlp_regressor`, `xgboost_regressor`, and `lstm_regressor`.
- `save_models`: toggle persistence of trained estimators.

Add more configs (e.g., `configs/<experiment>.yaml`) to sweep different hyperparameters or estimators—the main function will automatically pick them up once the model is registered in `models/__init__.py`.

---

## Quick Start (All at Once)

```bash
chmod +x scripts/*.py run_all_linear.sh
make full   # runs normalize -> train_linear -> train_extra -> aggregate -> report
```

---

## Project Structure

```
scripts/
├── initial_data_exploration.py   # Explore raw CSV files
├── process_data_full.py          # Full processing with walk-forward validation
├── process_data_naive.py         # Simple baseline processing
├── full_data_exploration.py      # Explore processed data
├── build_pipeline.py             # Normalize data with StandardScaler
├── train_linear.py               # Ridge / LinearRegression baseline trainer
├── train_tree_nn.py              # RandomForest / GradientBoosting / MLP trainer
├── aggregate_metrics.py          # Combine metrics JSON files into CSV summary
├── report_plots.py               # Generate RMSE comparison plot + markdown
└── grid_search.py                # Hyperparameter sweeps per window/fold

data/
├── raw/                          # Raw Kaggle data
├── naive_processed/              # Simple processed data
├── full_processed/               # Walk-forward validation data
└── normalized/                   # Normalized data for model training

experiments/                       # Model training scripts (to be added)
reports/                          # Results and analysis
```

## Next Steps: Model Training

Once normalized data is ready (`data/normalized/`), you can:

1. **Train baseline models** (Linear Regression, Random Forest, etc.)
   - `./run_all_linear.sh` or `make train_linear`
   - `python scripts/train_tree_nn.py --model rf|gbr|mlp ...`
2. **Use cross-validation / grid search** to tune hyperparameters
   - `python scripts/grid_search.py --model ridge --param-grid '{"alpha":[0.1,1,10]}'`
3. **Aggregate and visualize metrics**:
   - `python scripts/aggregate_metrics.py`
   - `python scripts/report_plots.py`
4. **Generate graphs** showing:
   - Parameter tuning results
   - Model performance on eval set
   - Predictions vs actual prices
5. **Compare models** and document results using `reports/model_report.md`

Use the normalized data structure:
```python
import pandas as pd
from pathlib import Path

window_size = 3
fold = 0

X_train = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/X_train.csv")
y_train = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/y_train.csv")
X_test = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/X_test.csv")
y_test = pd.read_csv(f"data/normalized/window_{window_size}/fold_{fold}/y_test.csv")
X_eval = pd.read_csv(f"data/normalized/window_{window_size}/X_eval.csv")
y_eval = pd.read_csv(f"data/normalized/window_{window_size}/y_eval.csv")

# Your model training here...
```

## Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - ML preprocessing and models
- `matplotlib` - Plotting and visualization
- `xgboost` - Gradient-boosted trees (macOS users: `brew install libomp` if you hit runtime loader errors)
- `torch` - Needed for `lstm_regressor` (install a wheel compatible with your Python version or build from source)
- `python-dateutil`, `pytz`, `tzdata` - Date/time handling

See `requirements.txt` for specific versions.

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'pandas'`
- **Solution:** Ensure conda environment is activated and packages are installed
  ```bash
  conda activate cmpe257
  conda install pandas -y
  ```

**Issue:** Scripts take too long
- **Solution:** This is normal for large datasets. Process data once, then reuse normalized sets.

**Issue:** Out of memory errors
- **Solution:** The full dataset is large. If needed, modify scripts to process in batches or reduce data sample.

## Authors & Notes
- Cale Payson, Sonali Lonkar, Ramya Gopalaswamy, Pranav Sehgal
- Repository: CMPE 257 Course Project
- Data source: Kaggle
- Processing: Custom walk-forward validation pipeline
- Framework: scikit-learn for preprocessing and scaling
