import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ── Step 1: Create Time Series Data ───────
# We'll predict sine wave — simple but perfect
# for understanding sequence models
print("=== Time Series Prediction with LSTM ===")

# Generate sine wave
timesteps = 1000
t         = np.linspace(0, 100, timesteps)
data      = np.sin(t) + 0.1 * np.random.randn(timesteps)
# plt.figure(figsize=(12, 3))
# plt.plot(t[:100], data[:100], color='blue')
# plt.title("Sine Wave Time Series (first 100 steps)")
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.savefig("time_series.png")
# plt.show()
# print("Time series plotted!")

# ── Step 2: Prepare Sequence Data ─────────
# Scale data to [0, 1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(
    data.reshape(-1, 1)).flatten()
# Create sequences
# Use past 20 steps to predict next step
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
seq_length = 20
X, y       = create_sequences(data_scaled, seq_length)
print(f"\nSequence shape : {X.shape}")
print(f"Label shape    : {y.shape}")
print(f"Example: use steps 0-19 to predict step 20")
# Train/test split
split    = int(0.8 * len(X))
X_train  = torch.FloatTensor(X[:split]).unsqueeze(-1)
X_test   = torch.FloatTensor(X[split:]).unsqueeze(-1)
y_train  = torch.FloatTensor(y[:split])
y_test   = torch.FloatTensor(y[split:])
print(f"\nX_train shape  : {X_train.shape}")
print(f"(samples, seq_length, features)")
# ── Step 3: Build Models ───────────────────
# Simple RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32,
                 num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        out    = self.fc(out[:, -1, :])  # last timestep
        return out.squeeze()
# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32,
                 num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.fc(out[:, -1, :])  # last timestep
        return out.squeeze()
# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32,
                 num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out    = self.fc(out[:, -1, :])
        return out.squeeze()
# ── Step 4: Train Function ────────────────
def train_model(model, model_name, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses    = []
    print(f"\nTraining {model_name}...")
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss   = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Loss: {loss.item():.6f}")
   # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).numpy()
        y_true    = y_test.numpy()
        mse       = mean_squared_error(y_true, test_pred)
        rmse      = np.sqrt(mse)
    print(f"{model_name} RMSE: {rmse:.4f}")
    return losses, test_pred
# ── Step 5: Train All 3 Models ────────────
rnn_model  = SimpleRNN()
lstm_model = LSTMModel()
gru_model  = GRUModel()
rnn_losses,  rnn_pred  = train_model(rnn_model,  'RNN')
lstm_losses, lstm_pred = train_model(lstm_model, 'LSTM')
gru_losses,  gru_pred  = train_model(gru_model,  'GRU')
# ── Step 6: Visualize Predictions ─────────
y_test_np = y_test.numpy()
# Inverse transform to original scale
def inverse(arr):
    return scaler.inverse_transform(
        arr.reshape(-1, 1)).flatten()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Loss comparison
axes[0].plot(rnn_losses,  label='RNN',  color='red')
axes[0].plot(lstm_losses, label='LSTM', color='blue')
axes[0].plot(gru_losses,  label='GRU',  color='green')
axes[0].set_title("Training Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].legend()
# Prediction comparison
actual = inverse(y_test_np)
axes[1].plot(actual[:100],
             label='Actual', color='black', linewidth=2)
axes[1].plot(inverse(rnn_pred[:100]),
             label='RNN',    color='red',   alpha=0.7)
axes[1].plot(inverse(lstm_pred[:100]),
             label='LSTM',   color='blue',  alpha=0.7)
axes[1].plot(inverse(gru_pred[:100]),
             label='GRU',    color='green', alpha=0.7)
axes[1].set_title("Predictions vs Actual (first 100)")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Value")
axes[1].legend()
plt.suptitle("RNN vs LSTM vs GRU Comparison",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
# ── Step 7: Final Summary ─────────────────
print("\n=== Final Summary ===")
print("RNN  → Simple, fast, forgets long sequences")
print("LSTM → Complex, slow, remembers long sequences")
print("GRU  → Simple as RNN, memory like LSTM")
print("\nFor most sequence tasks → use LSTM or GRU!")





print("All is good")


