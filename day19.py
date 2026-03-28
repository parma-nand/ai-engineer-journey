import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("=" * 50)
print("Day 19 — PyTorch vs TensorFlow Comparison")
print("=" * 50)
print(f"PyTorch version    : {torch.__version__}")
print(f"TensorFlow version : {tf.__version__}")
# ── Step 1: Load & Prepare Same Dataset ─────
# Using same Titanic dataset as Day 18
# So we can compare RESULTS fairly
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df  = pd.read_csv(url)
df['Age']      = df['Age'].fillna(df['Age'].mean())
df['Fare']     = df['Fare'].fillna(df['Fare'].mean())
df['Sex']      = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X = df[['Age', 'Sex', 'Pclass', 'Fare',
        'Embarked', 'SibSp', 'Parch']].values
y = df['Survived'].values
scaler = StandardScaler()
X      = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
# ── Step 2: Build Neural Network in PyTorch ──
print("\n=== PyTorch Neural Network ===")
class PyTorchNN(nn.Module):
    def __init__(self):
        super(PyTorchNN, self).__init__()
        self.fc1      = nn.Linear(7, 32)
        self.relu1    = nn.ReLU()
        self.bn1      = nn.BatchNorm1d(32)
        self.drop1    = nn.Dropout(0.3)
        self.fc2      = nn.Linear(32, 16)
        self.relu2    = nn.ReLU()
        self.bn2      = nn.BatchNorm1d(16)
        self.drop2    = nn.Dropout(0.2)
        self.fc3      = nn.Linear(16, 1)
        self.sigmoid  = nn.Sigmoid()
    def forward(self, x):
        x = self.drop1(self.bn1(self.relu1(self.fc1(x))))
        x = self.drop2(self.bn2(self.relu2(self.fc2(x))))
        x = self.sigmoid(self.fc3(x))
        return x
# Convert data to PyTorch tensors
X_train_pt = torch.FloatTensor(X_train)
y_train_pt = torch.FloatTensor(y_train)
X_test_pt  = torch.FloatTensor(X_test)
y_test_pt  = torch.FloatTensor(y_test)
# Initialize model, loss, optimizer
pt_model   = PyTorchNN()
criterion  = nn.BCELoss()
optimizer  = torch.optim.Adam(
    pt_model.parameters(), lr=0.001)
# Manual training loop
pt_losses      = []
pt_accuracies  = []
print("Training PyTorch model...")
for epoch in range(50):
    pt_model.train()
    # Forward pass
    y_pred = pt_model(X_train_pt).squeeze()
    loss   = criterion(y_pred, y_train_pt)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Track accuracy
    predicted = (y_pred >= 0.5).float()
    accuracy  = (predicted == y_train_pt).float().mean()
    pt_losses.append(loss.item())
    pt_accuracies.append(accuracy.item() * 100)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | "
              f"Loss: {loss.item():.4f} | "
              f"Accuracy: {accuracy.item()*100:.1f}%")
# Evaluate PyTorch model
pt_model.eval()
with torch.no_grad():
    y_pred_prob = pt_model(X_test_pt).squeeze()
    y_pred_pt   = (y_pred_prob >= 0.5).float()
    pt_accuracy = accuracy_score(
        y_test, y_pred_pt.numpy())
print(f"\nPyTorch Test Accuracy: {pt_accuracy*100:.1f}%")
# ── Step 3: Build Same Network in TensorFlow ─
print("\n=== TensorFlow/Keras Neural Network ===")
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        32, activation='relu', input_shape=(7,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
tf_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)
print("Training TensorFlow model...")
history = tf_model.fit(
    X_train, y_train,
    epochs           = 50,
    batch_size       = 32,
    validation_split = 0.2,
    verbose          = 0       # silent
)
_, tf_accuracy = tf_model.evaluate(
    X_test, y_test, verbose=0)
print(f"TensorFlow Test Accuracy: {tf_accuracy*100:.1f}%")
# ── Step 4: Compare Results ──────────────────
print("\n=== Final Comparison ===")
print(f"PyTorch    Accuracy : {pt_accuracy*100:.1f}%")
print(f"TensorFlow Accuracy : {tf_accuracy*100:.1f}%")
print(f"\nSame architecture → Similar results!")
print(f"Difference is just SYNTAX not performance!")
# ── Step 5: Side by Side Syntax Comparison ───
print("\n=== Syntax Comparison ===")
print("""
TASK               PYTORCH                    TENSORFLOW
──────────────────────────────────────────────────────────
Define Layer     : nn.Linear(7,32)         → Dense(32)
Activation       : nn.ReLU()               → activation='relu'
Dropout          : nn.Dropout(0.3)         → Dropout(0.3)
BatchNorm        : nn.BatchNorm1d(32)      → BatchNormalization()
Loss             : nn.BCELoss()            → binary_crossentropy
Optimizer        : optim.Adam(lr=0.001)    → Adam(lr=0.001)
Training         : manual loop             → model.fit()
Backprop         : loss.backward()         → automatic
Weight Update    : optimizer.step()        → automatic
Zero Gradients   : optimizer.zero_grad()   → automatic
Evaluate         : model.eval()            → model.evaluate()
Predict          : model(X_test)           → model.predict()
No Gradient      : torch.no_grad()         → automatic
Save Model       : torch.save()            → model.save()
""")
# ── Step 6: Visualize Comparison ────────────
tf_losses      = history.history['loss']
tf_accuracies  = [a * 100 for a in
                  history.history['accuracy']]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Loss comparison
axes[0].plot(pt_losses,
             label='PyTorch',
             color='blue')
axes[0].plot(tf_losses,
             label='TensorFlow',
             color='orange')
axes[0].set_title("Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
# Accuracy comparison
axes[1].plot(pt_accuracies,
             label='PyTorch',
             color='blue')
axes[1].plot(tf_accuracies,
             label='TensorFlow',
             color='orange')
axes[1].set_title("Accuracy Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy %")
axes[1].legend()
plt.suptitle(
    "PyTorch vs TensorFlow — Same Model, Same Data",
    fontsize=13, fontweight='bold')
plt.tight_layout()
# plt.savefig("pytorch_vs_tensorflow.png")
plt.show()
print("\n✅ Day 19 Complete!")
print("Chart saved as pytorch_vs_tensorflow.png")
