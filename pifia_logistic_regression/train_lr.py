import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

LOG_TRAINING = True # Set to True to enable logging of training output

# --- Model Hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
FEATURE_COLS = [f"f{i}" for i in range(64)]

# --- Logging Setup ---
if LOG_TRAINING:
    import logging

    # Log training output to a file
    logfile_path = "model_training.log"
    logging.basicConfig(
        filename=logfile_path,
        filemode='w',
        level=logging.INFO,
        format='%(message)s'
    )

    class LoggerWriter:
        def __init__(self, logger_func):
            self.logger_func = logger_func

        def write(self, message):
            if message.strip():
                self.logger_func(message.strip())

        def flush(self):
            pass

    sys.stdout = LoggerWriter(logging.info)

# --- Dataset Loader ---
def load_dataset(path):
    df = pd.read_csv(path)
    df = df[df["localization"] != "unknown"]
    df["localization"] = df["localization"].astype("category")
    return df

# --- Dataset Preparator ---
def prepare_data(df, common_filenames):
    df = df[df["filepath"].apply(os.path.basename).isin(common_filenames)]
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df["localization"].cat.codes.values, dtype=torch.long)
    num_classes = len(df["localization"].cat.categories)
    return X, y, num_classes

# --- Model ---
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# --- Training Function ---
def train_model(X, y, num_classes, save_path):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LogisticRegressionTorch(X.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    return model

# --- Main ---
if __name__ == "__main__":
    print("🔄 Loading datasets")
    df1 = load_dataset("pifia_feature_sets/original_crops_train.csv")
    df2 = load_dataset("pifia_feature_sets/masked_crops_train.csv")

    print(f"Dataset 1: {len(df1)} samples, {len(df1['localization'].cat.categories)} classes")

    # Filter common filenames
    filenames1 = set(df1["filepath"].apply(os.path.basename))
    filenames2 = set(df2["filepath"].apply(os.path.basename))
    common_files = filenames1 & filenames2

    X1, y1, num_classes1 = prepare_data(df1, common_files)
    X2, y2, num_classes2 = prepare_data(df2, common_files)

    print("\nTraining model on Dataset 1 (Original Crops Model)")
    model1 = train_model(X1, y1, num_classes1, "trained_models/original_crops_model.pth")

    print("\nTraining model on Dataset 2 (Masked Crops Model)")
    model2 = train_model(X2, y2, num_classes2, "trained_models/masked_crops_model.pth")
