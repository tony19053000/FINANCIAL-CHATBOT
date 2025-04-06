import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess crypto data
def load_and_preprocess_crypto_data(filepath, sequence_length=60):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['timestamp'], format="%d-%m-%Y %H:%M")
    df.sort_values(['coin_name', 'date'], inplace=True)

    features = ['price', 'volume', 'market_cap']
    grouped = df.groupby('coin_name')

    coin_data = {}
    for coin, group in grouped:
        group = group[['date'] + features]
        if len(group) < sequence_length + 1:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(group[features])

        X, y, dates = [], [], []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i])
            y.append(scaled[i])  # Predicting all features
            dates.append(group.iloc[i]['date'])

        coin_data[coin] = {
            'X': np.array(X),
            'y': np.array(y),
            'dates': dates,
            'scaler': scaler
        }
    return coin_data

# Define the LSTM model
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Train the model

def train_crypto_model(coin, data):
    X_tensor = torch.tensor(data['X'], dtype=torch.float32)
    y_tensor = torch.tensor(data['y'], dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryptoLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n📊 Training crypto model for {coin}")
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"{coin} - Epoch {epoch+1}/10 - Loss: {epoch_loss:.6f}")

    print(f"✅ Finished training for {coin}")
    torch.save(model.state_dict(), f"models/{coin}_crypto_lstm_model.pth")

    # Plot predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor.to(device)).cpu().numpy()
    actual = y_tensor.numpy()

    plt.figure(figsize=(10, 5))
    for i, label in enumerate(['Price', 'Volume', 'Market Cap']):
        plt.plot(data['dates'], actual[:, i], label=f'Actual {label}', alpha=0.6)
        plt.plot(data['dates'], predictions[:, i], label=f'Predicted {label}', alpha=0.6)

    plt.title(f"{coin} - LSTM Predictions")
    plt.xlabel("Date")
    plt.ylabel("Scaled Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"models/{coin}_crypto_prediction_plot.png")
    plt.close()

# Run training
if __name__ == "__main__":
    filepath = "data/reshaped_crypto_data.csv"
    crypto_data = load_and_preprocess_crypto_data(filepath)

    for coin, data in crypto_data.items():
        train_crypto_model(coin, data)