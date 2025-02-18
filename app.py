# Imports
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------- # Configuration Section ---------------- #
np.random.seed(42)
torch.manual_seed(42)

# Hyperparameters
INPUT_SIZE = 3
HIDDEN_1 = 64
HIDDEN_2 = 32
HIDDEN_3 = 16
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 32

# File Paths
MODEL_SAVE_PATH = "models/health_score_model.pth"
PLOTS_SAVE_DIR = "plots"


# ---------------- # Dataset Preparation ---------------- #
def generate_dataset(num_samples=1000):
    """
    Generate synthetic health predictor dataset with three features and a health score.

    Returns:
        pd.DataFrame: A DataFrame with columns [steps, calories_burned, sleep_hours, health_score].
    """
    steps = np.random.randint(1000, 20000, num_samples)
    calories_burned = np.random.uniform(1200, 4500, num_samples)
    sleep_hours = np.random.uniform(4, 12, num_samples)

    health_score = (
            0.3 * (steps / 20000) * 100 +
            0.4 * ((calories_burned - 1200) / (4500 - 1200)) * 100 +
            0.3 * (sleep_hours / 12) * 100
    )
    health_score += np.random.normal(0, 5, num_samples)

    df = pd.DataFrame({
        'steps': steps,
        'calories_burned': calories_burned,
        'sleep_hours': sleep_hours,
        'health_score': health_score
    })

    print("Correlation Matrix:")
    print(df.corr())  # Print relationships for exploration
    return df


def preprocess_data(df):
    """
    Preprocess the dataset: feature scaling and train-test split.

    Args:
        df (pd.DataFrame): Original dataset.

    Returns:
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, StandardScaler: Processed tensors for PyTorch.
    """
    X = df.drop('health_score', axis=1)
    y = df['health_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler


# ---------------- # Dataset and DataLoader ---------------- #
class HealthDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ---------------- # Model Definition ---------------- #
class HealthScoreNN(nn.Module):
    def __init__(self, input_size):
        super(HealthScoreNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, HIDDEN_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_3, 1)
        )

    def forward(self, x):
        return self.fc(x)


# ---------------- # Training Loop ---------------- #
def train_model(model, train_loader, test_loader, epochs, criterion, optimizer):
    """
    Train a PyTorch model using specified DataLoaders and hyperparameters.
    """
    epoch_loss = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        epoch_loss.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # Model evaluation on test data
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f" - Test Loss: {avg_test_loss:.4f}")

    return epoch_loss


# ---------------- # Main Script ---------------- #
if __name__ == "__main__":
    # Data preparation
    df = generate_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Dataset and DataLoader
    train_dataset = HealthDataset(X_train, y_train)
    test_dataset = HealthDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Criterion, Optimizer
    model = HealthScoreNN(INPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train the model
    losses = train_model(model, train_loader, test_loader, EPOCHS, criterion, optimizer)

    # Save the model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model, MODEL_SAVE_PATH)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
