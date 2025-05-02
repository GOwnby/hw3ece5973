
# Libaries required
#!pip install yfinance pandas numpy torch torchmetrics scikit-learn pytorch-lightning tqdm optuna mamba-ssm causal-conv1d>=1.1.0 matplotlib -q

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchmetrics
from tqdm.notebook import tqdm
import copy
import os
import matplotlib.pyplot as plt # Import for plotting

# Try importing Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba-ssm package not found. Mamba model type will not be available.")
    MAMBA_AVAILABLE = False


# Data Parameters
FEATURE_STOCKS = ['tsla', 'meta', 'nvda', 'amzn', 'nflx', 'gbtc', 'gdx', 'intc', 'dal', 'c', 'goog', 'aapl', 'msft', 'ibm', 'hp', 'orcl', 'sap', 'crm', 'hubs', 'twlo']
PREDICT_STOCK = 'msft'
START_DATE = '2020-01-01'
END_DATE = None

# Lookback Window
DAYS_WINDOW = 10 # Sequence length

# Splitting & Loading Parameters
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count() // 2
RANDOM_SEED = 42

# Model & Training Parameters
# Choose 'MLP', 'CNN', 'LSTM', 'Attention', 'Mamba'
# Set MODEL_TYPE to the *student* model architecture for distillation
MODEL_TYPE = 'LSTM' # Defaulting to LSTM for HW3 start
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Model Specific Hyperparameters
# These define the STUDENT model when distillation is active
# MLP
MLP_HIDDEN_1 = 8
MLP_HIDDEN_2 = 4
# CNN
CNN_OUT_CHANNELS_1 = 8
CNN_KERNEL_SIZE_1 = 3
CNN_LINEAR_FEATURES = 16
# LSTM
LSTM_HIDDEN_SIZE = 32 # Example size for LSTM
LSTM_NUM_LAYERS = 1
# Attention
ATTN_EMBED_DIM = len(FEATURE_STOCKS) # Match feature dim by default
ATTN_NUM_HEADS = 4
ATTN_LINEAR_FEATURES = 16
# Mamba
MAMBA_DIM = len(FEATURE_STOCKS) # Match feature dim by default
MAMBA_LAYER = 1

# Knowledge Distillation Parameters
ENABLE_DISTILLATION = False # Set to True to activate distillation training
DISTILLATION_ALPHA = 0.5 # Weight for distillation loss (0=no distillation, 1=only distillation)
# Set this to the path of your trained teacher checkpoint
TEACHER_MODEL_PATH = None # e.g., 'checkpoints/stock-LSTM-epoch=XX-val_loss=Y.YY.ckpt'
# Set this to the class of your teacher model
TEACHER_MODEL_CLASS = None # e.g., StockLSTM, StockAttention, StockCNN, StockMLP

# Fetch Data
def get_price(tick, start, end):
    try:
        data = yf.Ticker(tick).history(start=start, end=end)['Close']
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
        return data.dropna()
    except Exception as e:
        print(f"Error fetching data for {tick}: {e}")
        return pd.Series(dtype=float)

def get_prices(tickers, start, end):
    df = pd.DataFrame()
    print("Fetching data...")
    for s in tqdm(tickers):
        price_data = get_price(s, start, end)
        if not price_data.empty:
            df[s] = price_data
        else:
            print(f"Warning: No data fetched for {s}. Skipping.")
    df = df.ffill().bfill()
    df = df.dropna(subset=[PREDICT_STOCK])
    print("Data fetching complete.")
    return df

# Construct Stock Data Class
class StockDataset(Dataset):
    def __init__(self, X, Y, days)
        if X.shape[0] != Y.shape[0]:
             raise ValueError(f"X and Y must have the same number of samples (X shape: {X.shape[0]}, Y shape: {Y.shape[0]})")
        self.X = X.astype(np.float32)
        self.Y = Y.reshape(-1).astype(np.float32)
        self.days = days
        self.num_samples = len(self.Y) - self.days
        if self.num_samples <= 0:
             raise ValueError(f"Not enough data ({len(self.Y)} points) for the given lookback window ({self.days})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x_window = self.X[index : index + self.days, :]
        y_target = self.Y[index + self.days]
        return torch.from_numpy(x_window), torch.tensor(y_target)

# PyTorch Lightning DataModule (Fetch, Scale, Split)
class StockDataModule(pl.LightningDataModule):
    def __init__(self, feature_stocks, predict_stock, start_date, end_date, days_window,
                 train_ratio=0.7, val_ratio=0.15, batch_size=32, num_workers=4, seed=42):
        super().__init__()
        self.save_hyperparameters()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.num_features = len(feature_stocks)
        # Keep placeholders for dates, especially test_target_dates
        self.train_dates_raw = None
        self.val_dates_raw = None
        self.test_dates_raw = None
        self.test_target_dates = None # Dates for the test set targets

    def prepare_data(self):
        get_prices(self.hparams.feature_stocks, self.hparams.start_date, self.hparams.end_date)

    def setup(self, stage=None):
        all_prices_df = get_prices(self.hparams.feature_stocks, self.hparams.start_date, self.hparams.end_date)
        feature_data = all_prices_df[self.hparams.feature_stocks].values
        target_data = all_prices_df[[self.hparams.predict_stock]].values
        dates = all_prices_df.index

        n_total = len(target_data)
        n_train = int(n_total * self.hparams.train_ratio)
        n_val = int(n_total * self.hparams.val_ratio)

        X_train_raw, Y_train_raw = feature_data[:n_train], target_data[:n_train]
        X_val_raw, Y_val_raw = feature_data[n_train : n_train + n_val], target_data[n_train : n_train + n_val]
        X_test_raw, Y_test_raw = feature_data[n_train + n_val :], target_data[n_train + n_val :]
        # Split dates accordingly (still needed for test plot)
        self.train_dates_raw = dates[:n_train]
        self.val_dates_raw = dates[n_train : n_train + n_val]
        self.test_dates_raw = dates[n_train + n_val :]

        print("Fitting scalers on training data...")
        self.feature_scaler.fit(X_train_raw)
        self.target_scaler.fit(Y_train_raw)

        X_train_scaled = self.feature_scaler.transform(X_train_raw)
        Y_train_scaled = self.target_scaler.transform(Y_train_raw)
        X_val_scaled = self.feature_scaler.transform(X_val_raw)
        Y_val_scaled = self.target_scaler.transform(Y_val_raw)
        X_test_scaled = self.feature_scaler.transform(X_test_raw)
        Y_test_scaled = self.target_scaler.transform(Y_test_raw)
        print("Scaling complete.")

        print("Creating datasets...")
        # Pass only X, Y, days to StockDataset
        self.train_dataset = StockDataset(X_train_scaled, Y_train_scaled, self.hparams.days_window)
        self.val_dataset = StockDataset(X_val_scaled, Y_val_scaled, self.hparams.days_window)
        self.test_dataset = StockDataset(X_test_scaled, Y_test_scaled, self.hparams.days_window)
        print("Datasets created.")
        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}, Test samples: {len(self.test_dataset)}")

        # Store the dates corresponding to the *target* values in the test set for plotting
        # These are the dates from the original test split, offset by the window size
        self.test_target_dates = self.test_dates_raw[self.hparams.days_window:].tolist() # Ensure it's a list


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, persistent_workers=True if self.hparams.num_workers > 0 else False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, persistent_workers=True if self.hparams.num_workers > 0 else False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, persistent_workers=True if self.hparams.num_workers > 0 else False)

# PyTorch Lightning Model Base Class
class StockForecastBase(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters('learning_rate')
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_step_outputs = []
        self.teacher_model = None # Initialize teacher model placeholder

    def setup_teacher(self, teacher_path, teacher_class):
        """Loads the teacher model from a checkpoint."""
        if teacher_path and os.path.exists(teacher_path) and teacher_class:
            try:
                print(f"Loading teacher model ({teacher_class.__name__}) from {teacher_path}")
                self.teacher_model = teacher_class.load_from_checkpoint(teacher_path)
                self.teacher_model.eval() # Set teacher to evaluation mode
                self.teacher_model.freeze() # Freeze teacher weights
                print("Teacher model loaded and frozen.")
            except Exception as e:
                print(f"Error loading teacher model: {e}")
                self.teacher_model = None
        else:
            print("Warning: Teacher model path or class not provided/invalid. No distillation.")
            self.teacher_model = None

    def training_step(self, batch, batch_idx):
        x, y = batch # Batch no longer contains dates
        y_hat_student = self(x) # Student prediction

        # Standard supervised loss
        loss_pred = nn.functional.mse_loss(y_hat_student.squeeze(), y)

        # Knowledge Distillation Loss (if teacher is available)
        loss_distill = 0.0
        if self.teacher_model is not None and ENABLE_DISTILLATION and DISTILLATION_ALPHA > 0:
            # Ensure teacher model is on the same device as input
            if self.teacher_model.device != x.device:
                 self.teacher_model.to(x.device)

            with torch.no_grad():
                 y_hat_teacher = self.teacher_model(x).squeeze()

            # Use MSE between student and teacher outputs as distillation loss
            loss_distill = nn.functional.mse_loss(y_hat_student.squeeze(), y_hat_teacher)
            self.log('distil_loss', loss_distill, on_step=False, on_epoch=True, logger=True)

            # Combine losses
            loss = (1.0 - DISTILLATION_ALPHA) * loss_pred + DISTILLATION_ALPHA * loss_distill
        else:
            loss = loss_pred # Use only prediction loss if no distillation

        self.train_mse(y_hat_student.squeeze(), y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_pred_loss', loss_pred, on_step=False, on_epoch=True, logger=True)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y)
        self.val_mse(y_hat.squeeze(), y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y)
        self.test_mse(y_hat.squeeze(), y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mse_scaled', self.test_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        output_data = {'preds': y_hat.detach().cpu(), 'targets': y.detach().cpu()}
        self.test_step_outputs.append(output_data)
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("Warning: No test outputs recorded.")
            return

        # Collate data from all test batches
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).numpy()
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs]).numpy()
        self.test_step_outputs.clear() # Free memory

        # Get dates from DataModule 
        all_dates = None
        if hasattr(self.trainer.datamodule, 'test_target_dates'):
             all_dates = self.trainer.datamodule.test_target_dates
             if len(all_dates) != len(all_preds):
                  print(f"Warning: Mismatch between stored test dates ({len(all_dates)}) and predictions ({len(all_preds)}). Plotting might be incorrect.")
                  all_dates = None # Don't plot if mismatch
        else:
             print("Warning: test_target_dates not found in datamodule. Cannot plot dates.")


        # Inverse transform and calculate final MSE
        if hasattr(self.trainer.datamodule, 'target_scaler'):
            target_scaler = self.trainer.datamodule.target_scaler
            if all_preds.ndim == 1: all_preds = all_preds.reshape(-1, 1)
            preds_rescaled = target_scaler.inverse_transform(all_preds)
            targets_rescaled = target_scaler.inverse_transform(all_targets.reshape(-1, 1))

            final_mse = np.mean((targets_rescaled - preds_rescaled) ** 2)
            self.log('test_mse_original_scale', final_mse, logger=True)
            print(f"\nFinal Test MSE (Original Price Scale): {final_mse:.4f}")
            if final_mse < 20: print("Test MSE is less than 20.")
            elif final_mse < 120: print("Test MSE is less than 120.")
            else: print("Test MSE is above 120.")

            # Plot Actual vs Predicted
            if all_dates is not None:
                try:
                    plot_dates = pd.to_datetime(all_dates)
                except Exception as e:
                    print(f"Could not convert dates for plotting: {e}")
                    plot_dates = None

                if plot_dates is not None:
                    plt.figure(figsize=(12, 6))
                    plt.plot(plot_dates, targets_rescaled.flatten(), label='Actual Price', marker='.', linestyle='-')
                    plt.plot(plot_dates, preds_rescaled.flatten(), label='Predicted Price', marker='.', linestyle='--')
                    plt.title(f'{MODEL_TYPE} - Actual vs Predicted Prices (Test Set)')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

        else:
            print("Warning: Target scaler not found. Cannot calculate MSE on original scale or plot.")


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# MLP Model
class StockMLP(StockForecastBase):
    def __init__(self, input_features, days_window, hidden_size_1=128, hidden_size_2=64, dropout_rate=0.2, learning_rate=0.001):
        super().__init__(learning_rate)
        self.save_hyperparameters()
        self.input_size = input_features * days_window
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_size, self.hparams.hidden_size_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.hparams.dropout_rate)
        self.fc2 = nn.Linear(self.hparams.hidden_size_1, self.hparams.hidden_size_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.hparams.dropout_rate)
        self.fc3 = nn.Linear(self.hparams.hidden_size_2, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

# CNN Model
class StockCNN(StockForecastBase):
    def __init__(self, input_features, days_window, out_channels_1=32, kernel_size_1=3, linear_features=64, dropout_rate=0.2, learning_rate=0.001):
        super().__init__(learning_rate)
        self.save_hyperparameters()
        self.conv1 = nn.Conv1d(in_channels=self.hparams.input_features, out_channels=self.hparams.out_channels_1, kernel_size=self.hparams.kernel_size_1)
        self.relu1 = nn.ReLU()
        self.conv1_out_len = self.hparams.days_window - self.hparams.kernel_size_1 + 1
        self.flatten = nn.Flatten()
        self.fc1_in_features = self.hparams.out_channels_1 * self.conv1_out_len
        self.fc1 = nn.Linear(self.fc1_in_features, self.hparams.linear_features)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.fc2 = nn.Linear(self.hparams.linear_features, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.relu1(self.conv1(x))
        x = self.flatten(x)
        x = self.dropout(self.relu2(self.fc1(x)))
        x = self.fc2(x)
        return x

# LSTM Model
class StockLSTM(StockForecastBase):
    def __init__(self, input_features, hidden_size=32, num_layers=1, dropout_rate=0.2, learning_rate=0.001):
        super().__init__(learning_rate)
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size=self.hparams.input_features, hidden_size=self.hparams.hidden_size,
                            num_layers=self.hparams.num_layers, batch_first=True,
                            dropout=self.hparams.dropout_rate if self.hparams.num_layers > 1 else 0)
        self.linear = nn.Linear(self.hparams.hidden_size, 1)
        self.dropout = nn.Dropout(self.hparams.dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        out = self.dropout(lstm_out[:, -1, :]) 
        out = self.linear(out)
        return out

# Attention Model
class StockAttention(StockForecastBase):
    def __init__(self, input_features, days_window, embed_dim=32, num_heads=4, linear_features=16, dropout_rate=0.2, learning_rate=0.001):
        super().__init__(learning_rate)
        self.save_hyperparameters()
        if self.hparams.input_features != self.hparams.embed_dim:
             self.input_proj = nn.Linear(self.hparams.input_features, self.hparams.embed_dim)
        else:
             self.input_proj = nn.Identity()
        self.attention = nn.MultiheadAttention(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads,
                                               dropout=self.hparams.dropout_rate, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hparams.embed_dim)
        self.fc1_in_features = self.hparams.embed_dim
        self.fc1 = nn.Linear(self.fc1_in_features, self.hparams.linear_features)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.fc2 = nn.Linear(self.hparams.linear_features, 1)

    def forward(self, x):
        x = self.input_proj(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        out = x[:, -1, :] 
        out = self.dropout(self.relu1(self.fc1(out)))
        out = self.fc2(out)
        return out

# Mamba Model
class StockMamba(StockForecastBase):
    def __init__(self, input_features, days_window, d_model=32, mamba_layer=1, dropout_rate=0.2, learning_rate=0.001):
        super().__init__(learning_rate)
        if not MAMBA_AVAILABLE: raise ImportError("Mamba model selected, but mamba-ssm package is not installed.")
        self.save_hyperparameters()
        if self.hparams.input_features != self.hparams.d_model:
             self.input_proj = nn.Linear(self.hparams.input_features, self.hparams.d_model)
        else:
             self.input_proj = nn.Identity()
        self.mamba = Mamba(d_model=self.hparams.d_model, d_state=16, d_conv=4, expand=2)
        self.dropout = nn.Dropout(self.hparams.dropout_rate)
        self.fc = nn.Linear(self.hparams.d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        mamba_out = self.mamba(x)
        out = self.dropout(mamba_out[:, -1, :])
        out = self.fc(out)
        return out

# Parameter Calculation
def calculate_hw1_params():
    hw1_input = 10; hw1_hidden = 20; hw1_layers = 5; hw1_output = 1
    return (hw1_input + 1) * hw1_hidden + (hw1_hidden + 1) * hw1_hidden * (hw1_layers - 1) + (hw1_hidden + 1) * hw1_output

# Plotting Function
def plot_loss_curves(log_dir):
    """Plots training and validation loss from CSV logs."""
    try:
        metrics_path = os.path.join(log_dir, 'metrics.csv')
        if not os.path.exists(metrics_path):
            print(f"Warning: Metrics file not found at {metrics_path}")
            return
        metrics_df = pd.read_csv(metrics_path)

        # Extract epochs with validation loss and corresponding training loss
        val_epochs = metrics_df[metrics_df['val_loss'].notna()]['epoch']
        val_loss = metrics_df[metrics_df['val_loss'].notna()]['val_loss']

        # Find the step corresponding to the end of each validation epoch
        val_steps = metrics_df[metrics_df['val_loss'].notna()]['step'].tolist()

        # Get training loss recorded at or just before the validation step for alignment
        train_loss_aligned = []
        train_epochs_aligned = []

        train_loss_steps = metrics_df[metrics_df['train_loss'].notna()][['step', 'train_loss', 'epoch']]
        if not train_loss_steps.empty:
             for epoch, step in zip(val_epochs, val_steps):
                  relevant_train_steps = train_loss_steps[train_loss_steps['step'] <= step]
                  if not relevant_train_steps.empty:
                     last_train_step_data = relevant_train_steps.iloc[-1]
                     train_loss_aligned.append(last_train_step_data['train_loss'])
                     train_epochs_aligned.append(epoch)

        if not train_epochs_aligned:
             print("Warning: Could not align training loss for plotting.")
             train_epochs_all = metrics_df[metrics_df['train_loss'].notna()]['epoch']
             train_loss_all = metrics_df[metrics_df['train_loss'].notna()]['train_loss']
             if not train_epochs_all.empty:
                  print("Plotting available training losses (may not align perfectly with validation).")
                  train_epochs_aligned = train_epochs_all
                  train_loss_aligned = train_loss_all
             else:
                  return 


        plt.figure(figsize=(10, 5))
        plt.plot(train_epochs_aligned, train_loss_aligned, label='Training Loss', marker='.') 
        plt.plot(val_epochs, val_loss, label='Validation Loss', marker='.')
        plt.title(f'{MODEL_TYPE} - Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting loss curves: {e}")
        if 'metrics_path' in locals(): print(f"Attempted to read from: {metrics_path}")


# Training Execution
if __name__ == '__main__':
    pl.seed_everything(RANDOM_SEED, workers=True)

    data_module = StockDataModule(
        feature_stocks=FEATURE_STOCKS, predict_stock=PREDICT_STOCK,
        start_date=START_DATE, end_date=END_DATE, days_window=DAYS_WINDOW,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, seed=RANDOM_SEED
    )

    num_features = len(FEATURE_STOCKS)
    model = None # Student model

    # Instantiate the STUDENT model based on MODEL_TYPE and hyperparameters
    if MODEL_TYPE == 'MLP':
        model = StockMLP(input_features=num_features, days_window=DAYS_WINDOW, hidden_size_1=MLP_HIDDEN_1, hidden_size_2=MLP_HIDDEN_2, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
    elif MODEL_TYPE == 'CNN':
         if DAYS_WINDOW >= CNN_KERNEL_SIZE_1: model = StockCNN(input_features=num_features, days_window=DAYS_WINDOW, out_channels_1=CNN_OUT_CHANNELS_1, kernel_size_1=CNN_KERNEL_SIZE_1, linear_features=CNN_LINEAR_FEATURES, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
         else: raise ValueError("DAYS_WINDOW must be >= CNN_KERNEL_SIZE_1")
    elif MODEL_TYPE == 'LSTM':
        model = StockLSTM(input_features=num_features, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
    elif MODEL_TYPE == 'Attention':
         if ATTN_EMBED_DIM != num_features:
             print(f"Adjusting ATTN_EMBED_DIM to {num_features} to match input features for simpler setup.")
             ATTN_EMBED_DIM = num_features
         model = StockAttention(input_features=num_features, days_window=DAYS_WINDOW, embed_dim=ATTN_EMBED_DIM, num_heads=ATTN_NUM_HEADS, linear_features=ATTN_LINEAR_FEATURES, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
    elif MODEL_TYPE == 'Mamba':
        if MAMBA_AVAILABLE:
             if MAMBA_DIM != num_features:
                 print(f"Adjusting MAMBA_DIM to {num_features} to match input features for simpler setup.")
                 MAMBA_DIM = num_features
             model = StockMamba(input_features=num_features, days_window=DAYS_WINDOW, d_model=MAMBA_DIM, mamba_layer=MAMBA_LAYER, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
        else: raise RuntimeError("Mamba model selected, but mamba-ssm package is not available.")
    else:
        raise ValueError("Invalid MODEL_TYPE selected for student model.")

    print(f"\n# Student Model: {MODEL_TYPE} #")

    # Setup Teacher Model for Distillation if enabled
    if ENABLE_DISTILLATION:
        if TEACHER_MODEL_PATH is None or TEACHER_MODEL_CLASS is None:
            raise ValueError("TEACHER_MODEL_PATH and TEACHER_MODEL_CLASS must be set for distillation.")
        # Pass the actual class, not the string name
        model.setup_teacher(TEACHER_MODEL_PATH, TEACHER_MODEL_CLASS)
        print(f"Distillation enabled with alpha={DISTILLATION_ALPHA}")
    else:
        print("Distillation disabled.")


    hw1_params = calculate_hw1_params()
    student_params = model.count_parameters()
    print(f"HW1 FCNN Parameters (Approx): {hw1_params}")
    print(f"Student {MODEL_TYPE} Parameters: {student_params}")

    log_name = f"stock_forecast_{MODEL_TYPE}{'_distilled' if ENABLE_DISTILLATION else ''}"
    logger = CSVLogger("logs", name=log_name)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', dirpath='checkpoints/',
        filename=f'stock-{MODEL_TYPE}{"-distilled" if ENABLE_DISTILLATION else ""}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=1, mode='min', save_last=False)

    if torch.cuda.is_available(): accelerator, devices = 'gpu', 1
    else: accelerator, devices = 'cpu', 1

    trainer = pl.Trainer(
        max_epochs=EPOCHS, accelerator=accelerator, devices=devices,
        logger=logger, callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True)

    print("\n# Starting Training #")
    trainer.fit(model=model, datamodule=data_module)
    print("# Training Complete #")

    # --- Plotting Loss ---
    log_dir = logger.log_dir # Get the specific version directory
    plot_loss_curves(log_dir)

    print("\n# Starting Testing (using best model checkpoint) #")
    # Testing automatically triggers the on_test_epoch_end hook which includes plotting
    trainer.test(model=model, datamodule=data_module, ckpt_path='best')
    print("# Testing Complete #")

    print("\n# End of Script #")
