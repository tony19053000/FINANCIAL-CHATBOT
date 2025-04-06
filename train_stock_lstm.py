import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load the CSV files
def load_csv_files():
    folder_path = "data"
    file_names = ["NIFTY_50.csv", "SENSEX.csv", "S&P_500.csv", "stock_market_data_25y_full.csv"]

    dataframes = {}
    for file in file_names:
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        print(f"Loaded {file} with shape {df.shape}")
        dataframes[file] = df

    return dataframes

# Step 2: Preprocess the stock data with open, high, low, close, volume
def preprocess_stock_data(df):
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.sort_values(['company_name', 'date'], inplace=True)

    features = ['open', 'high', 'low', 'close', 'volume']
    df_grouped = df.groupby(['company_name', 'ticker'])
    company_data = {}
    history_window = 360

    for (company, ticker), group in df_grouped:
        group = group[['date'] + features]
        if len(group) < history_window + 1:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(group[features])

        X, y, dates = [], [], []
        for i in range(history_window, len(scaled)):
            X.append(scaled[i-history_window:i])
            y.append(scaled[i][3])  # Predicting 'close'
            dates.append(group.iloc[i]['date'])

        company_data[company] = {
            'X': np.array(X),
            'y': np.array(y),
            'dates': dates,
            'scaler': scaler,
            'ticker': ticker
        }
    return company_data

# LSTM Model Definition
class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Preprocess index data (no ticker)
def preprocess_index_data(df, sequence_length=360):
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df.sort_values("date", inplace=True)

    features = ["open", "high", "low", "close", "volume"]
    df = df[["date"] + features].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y, dates = [], [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length:i])
        y.append(scaled[i][3])  # Predicting 'close'
        dates.append(df.iloc[i]["date"])

    return {
        "X": np.array(X),
        "y": np.array(y),
        "dates": dates,
        "scaler": scaler
    }

# Train LSTM model for index
def train_index_model(name, df):
    print(f"\n📊 Training index model for {name}")
    data = preprocess_index_data(df)

    X_tensor = torch.tensor(data["X"], dtype=torch.float32)
    y_tensor = torch.tensor(data["y"], dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        print(f"{name} - Epoch {epoch+1}/10 - Loss: {epoch_loss:.6f}")

    print(f"✅ Finished training for {name}")
    torch.save(model.state_dict(), f"models/{name}_lstm_model.pth")

    # ✅ Batch-wise prediction to avoid OOM
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 64):
            batch = X_tensor[i:i+64].to(device)
            batch_preds = model(batch).cpu().numpy()
            predictions.extend(batch_preds)

    predictions = np.array(predictions).flatten()
    actual = y_tensor.numpy().flatten()

    # Plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(data["dates"], actual, label="Actual", alpha=0.7)
    plt.plot(data["dates"], predictions, label="Predicted", alpha=0.7)
    plt.title(f"{name} - LSTM Predictions vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Close Price (scaled)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"models/{name}_prediction_plot.png")
    plt.close()

# Step 3: Run the pipeline
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    dfs = load_csv_files()
    df = dfs["stock_market_data_25y_full.csv"]
    company_data = preprocess_stock_data(df)

    # Train LSTM models for each company
    for company, data in company_data.items():
        X_tensor = torch.tensor(data['X'], dtype=torch.float32)
        y_tensor = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(-1)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StockLSTM().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"✅ Training started for {company} ({data['ticker']})...")
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
            print(f"{company} - Epoch {epoch+1}/10 - Loss: {epoch_loss:.6f}")
        print(f"✅ Training complete for {company}.\n")

        torch.save(model.state_dict(), f"models/{company}_lstm_model.pth")

    # Train index models
    train_index_model("NIFTY_50", dfs["NIFTY_50.csv"])
    train_index_model("SENSEX", dfs["SENSEX.csv"])
    train_index_model("S&P_500", dfs["S&P_500.csv"])
