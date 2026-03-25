import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ── Step 1: Load MNIST Dataset ────────────
# MNIST = handwritten digits 0-9
# 60,000 training images, 10,000 test images
# Each image = 28x28 pixels, grayscale

transform = transforms.Compose([
    transforms.ToTensor(),              # convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # normalize to [-1, 1]
])
# Download and load training data
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
# Download and load test data
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(
    test_dataset,  batch_size=64, shuffle=False)
print(f"Training samples : {len(train_dataset)}")
print(f"Testing samples  : {len(test_dataset)}")
print(f"Image shape      : {train_dataset[0][0].shape}")
print(f"Number of classes: 10 (digits 0-9)")
# ── Step 2: Visualize Sample Images ───────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.suptitle("Sample MNIST Images", fontsize=14)
plt.tight_layout()
plt.savefig("mnist_samples.png")
plt.show()
print("Sample images saved!")
# ── Step 3: Build CNN ─────────────────────
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # 1 channel in, 32 filters out
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # image: 28x28 → 14x14
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # 32 channels in, 64 filters out
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # image: 14x14 → 7x7
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),              # 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)        # 10 classes (0-9)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
# Create model
model     = DigitCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("\nCNN Architecture:")
print(model)
# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")
# ── Step 4: Train CNN ─────────────────────
print("\n=== Training CNN ===")
train_losses = []
train_accs   = []
epochs       = 5      # 5 epochs is enough for MNIST
for epoch in range(epochs):
    model.train()
    running_loss    = 0.0
    correct         = 0
    total_samples   = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss    = criterion(outputs, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track accuracy
        running_loss  += loss.item()
        _, predicted   = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct       += (predicted == labels).sum().item()
        # Print every 200 batches
        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch {epoch+1} | "
                  f"Batch {batch_idx+1} | "
                  f"Loss: {loss.item():.4f}")
    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = correct / total_samples
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Accuracy: {epoch_acc*100:.1f}%")
# ── Step 5: Evaluate on Test Data ─────────
print("\n=== Evaluating on Test Data ===")
model.eval()
correct       = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs      = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct       += (predicted == labels).sum().item()
test_acc = correct / total_samples
print(f"Test Accuracy: {test_acc*100:.2f}%")
# ── Step 6: Visualize Results ─────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Training loss
axes[0].plot(range(1, epochs+1), train_losses,
             marker='o', color='blue')
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
# Training accuracy
axes[1].plot(range(1, epochs+1),
             [a*100 for a in train_accs],
             marker='o', color='green')
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy %")
plt.suptitle(f"CNN Training Results | "
             f"Test Accuracy: {test_acc*100:.1f}%",
             fontsize=13)
plt.tight_layout()
# plt.savefig("cnn_training.png")
plt.show()
# ── Step 7: Predict on Sample Images ──────
print("\n=== Sample Predictions ===")
model.eval()
sample_images, sample_labels = next(iter(test_loader))
with torch.no_grad():
    outputs      = model(sample_images[:10])
    _, predicted = torch.max(outputs, 1)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(sample_images[i].squeeze(), cmap='gray')
    color = 'green' if predicted[i] == sample_labels[i] else 'red'
    ax.set_title(f"Pred:{predicted[i].item()} "
                 f"True:{sample_labels[i].item()}",
                 color=color)
    ax.axis('off')
plt.suptitle("Green=Correct  Red=Wrong", fontsize=12)
plt.tight_layout()
# plt.savefig("cnn_predictions.png")
plt.show()
print("All charts saved!")

