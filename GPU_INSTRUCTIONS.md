# Running the ML Project on GPU

This guide explains how to run the stock market prediction project (all models: Linear, Polynomial, Random Forest, Gradient Boosting, MLP, XGBoost, LSTM) on a GPU for faster training. XGBoost and LSTM models will automatically use GPU acceleration when available. Since CUDA is not supported on macOS, we recommend using Google Colab (free) or a cloud GPU instance.

## Option 1: Google Colab (Recommended, Free)

### Step 1: Set Up Colab
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Create a new notebook or open an existing one.
3. Change the runtime to GPU:
   - Click **Runtime** > **Change runtime type**.
   - Select **Hardware accelerator** > **GPU** (T4 GPU recommended).
   - Click **Save**.

### Step 2: Clone or Upload the Project
- Clone the repo:
  ```
  !git clone https://github.com/calepayson/cmpe_257_project.git
  %cd cmpe_257_project
  ```
- Or upload files manually via the Files panel.

### Step 3: Install Dependencies
Run this in a cell:
```
!pip install torch torchvision xgboost scikit-learn pandas numpy matplotlib pyyaml
```

### Step 4: Run the Training
Execute:
```
!python -m src.main --config configs/all_models.yaml
```
- This will train all models (Linear, Polynomial, Random Forest, Gradient Boosting, MLP, XGBoost, LSTM) with cross-validation.
- XGBoost and LSTM will use GPU acceleration for much faster training.
- Expected time: 30-60 minutes (vs. hours on CPU).
- Results will be saved in `results/`.

### Step 5: Monitor and Download Results
- Check the output for progress.
- Download results from the Files panel or save to Google Drive.

## Option 2: Cloud GPU (AWS/Google Cloud/Azure)

1. Launch a Linux VM with NVIDIA GPU (e.g., AWS P3 with V100).
2. Install CUDA toolkit from NVIDIA.
3. Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
4. Clone the repo and run as above.

## Code Modifications for GPU Support

The code has been updated to auto-detect GPU. In `models/lstm.py`:
- Device is set to CUDA if available, else CPU.
- Model and data are moved to the device.

If running on CPU-only, it falls back automatically.

## Troubleshooting
- **Out of memory**: Reduce `batch_size` in `configs/xgb_lstm.yaml`.
- **No GPU detected**: Ensure runtime is set to GPU in Colab.
- **Import errors**: Reinstall packages.

For questions, check the project README or open an issue.
