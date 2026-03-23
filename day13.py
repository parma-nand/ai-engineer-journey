import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── Step 1: Load & Prepare ────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df  = pd.read_csv(url)

df['Age']      = df['Age'].fillna(df['Age'].mean())
df['Fare']     = df['Fare'].fillna(df['Fare'].mean())
df['Sex']      = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[['Age', 'Sex', 'Pclass', 'Fare', 'Embarked']].values
y = df['Survived'].values

scaler = StandardScaler()
X      = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test  = torch.FloatTensor(y_test)

print("Data ready!")
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Step 2: Improved Network ──────────────
# Added Dropout + BatchNorm vs Day 12
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),    # normalize layer output
            nn.ReLU(),
            nn.Dropout(0.3),       # drop 30% neurons

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),       # drop 20% neurons

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
    
    # ── Step 3: Train with Different Optimizers
def train_model(optimizer_name, epochs=100):
    model     = ImprovedNet()
    criterion = nn.BCELoss()

    # Create optimizer based on name
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=0.01)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=0.001)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=0.001)

    train_losses = []
    test_losses  = []

    for epoch in range(epochs):
        # ── Training mode ─────────────────
        model.train()
        y_pred = model(X_train).squeeze()
        loss   = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())


    for epoch in range(epochs):
        # ── Training mode ─────────────────
        model.train()
        y_pred = model(X_train).squeeze()
        loss   = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # ── Evaluation mode ───────────────
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test).squeeze()
            test_loss = criterion(test_pred, y_test)
            test_losses.append(test_loss.item())

        # Print every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"  {optimizer_name} | "
                  f"Epoch {epoch+1:3} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Test Loss: {test_loss.item():.4f}")

    # Final accuracy
    model.eval()
    with torch.no_grad():
        pred   = model(X_test).squeeze()
        labels = (pred >= 0.5).float()
        acc    = accuracy_score(
            y_test.numpy(), labels.numpy())

    print(f"\n{optimizer_name} Final Accuracy: {acc*100:.1f}%")
    return train_losses, test_losses, acc
# ── Step 4: Run All 3 Optimizers ──────────
print("\n=== SGD ===")
sgd_train, sgd_test, sgd_acc = train_model('SGD')

print("\n=== Adam ===")
adam_train, adam_test, adam_acc = train_model('Adam')

print("\n=== RMSprop ===")
rms_train, rms_test, rms_acc = train_model('RMSprop')

# ── Step 5: Compare Results ───────────────
print("\n=== Final Comparison ===")
print(f"SGD      Accuracy: {sgd_acc*100:.1f}%")
print(f"Adam     Accuracy: {adam_acc*100:.1f}%")
print(f"RMSprop  Accuracy: {rms_acc*100:.1f}%")

best = max(
    [('SGD', sgd_acc), ('Adam', adam_acc),
     ('RMSprop', rms_acc)],
    key=lambda x: x[1]
)
print(f"\nBest Optimizer: {best[0]} ({best[1]*100:.1f}%)")
# ── Step 6: Plot Training Loss ────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training loss
axes[0].plot(sgd_train,  label='SGD',     color='red')
axes[0].plot(adam_train, label='Adam',    color='blue')
axes[0].plot(rms_train,  label='RMSprop', color='green')
axes[0].set_title("Training Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
# Test loss
axes[1].plot(sgd_test,  label='SGD',     color='red')
axes[1].plot(adam_test, label='Adam',    color='blue')
axes[1].plot(rms_test,  label='RMSprop', color='green')
axes[1].set_title("Test Loss Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.suptitle("Optimizer Comparison", fontsize=14,
             fontweight='bold')
plt.tight_layout()
# plt.show()
print("\nChart saved!")

# ── Step 7: Dropout Experiment ────────────
print("\n=== Dropout Experiment ===")
print("Dropout during training: randomly zeros neurons")
print("Dropout during eval: all neurons active\n")

# Show dropout effect
test_dropout = nn.Dropout(0.5)
x = torch.ones(10)

# Training mode - dropout active
test_dropout.train()
print("Training mode (dropout ON) :")
print(test_dropout(x))

# Eval mode - dropout off
test_dropout.eval()
print("\nEval mode (dropout OFF):")
print(test_dropout(x))
# ── Step 8: Learning Rate Effect ──────────
print("\n=== Learning Rate Effect ===")
learning_rates = [0.1, 0.01, 0.001, 0.0001]

lr_results = {}
for lr in learning_rates:
    model     = ImprovedNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses    = []

    for epoch in range(50):
        model.train()
        y_pred = model(X_train).squeeze()
        loss   = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    lr_results[lr] = losses
    print(f"LR={lr:.4f} | Final Loss: {losses[-1]:.4f}")

# Plot learning rate effect
plt.figure(figsize=(8, 4))
for lr, losses in lr_results.items():
    plt.plot(losses, label=f'LR={lr}')
plt.title("Effect of Learning Rate on Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
print("Learning rate chart saved!")
            