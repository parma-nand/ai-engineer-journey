import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# ── Step 1: Load CIFAR-10 Dataset ─────────
# CIFAR-10 = 10 classes of real images
# airplane, car, bird, cat, deer,
# dog, frog, horse, ship, truck
# 60,000 images, 32x32 pixels, color (3 channels)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),      # ResNet needs 224x224
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # ImageNet mean
        std=[0.229, 0.224, 0.225]       # ImageNet std
    )
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform_test)
# Use subset for faster training
train_subset = torch.utils.data.Subset(
    train_dataset, range(2000))
test_subset  = torch.utils.data.Subset(
    test_dataset,  range(500))
train_loader = torch.utils.data.DataLoader(
    train_subset, batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(
    test_subset,  batch_size=32, shuffle=False)
classes = ['airplane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Training samples : {len(train_subset)}")
print(f"Testing samples  : {len(test_subset)}")
print(f"Classes          : {classes}")
# ── Step 2: Visualize Sample Images ───────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flatten()):
    img, label = train_dataset[i]
    # Denormalize for display
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(classes[label])
    ax.axis('off')
plt.suptitle("CIFAR-10 Sample Images", fontsize=14)
plt.tight_layout()
plt.savefig("cifar10_samples.png")
plt.show()
# ── Step 3: Load Pretrained ResNet ────────
print("\n=== Loading Pretrained ResNet18 ===")
model = models.resnet18(pretrained=True)
print("ResNet18 loaded — pretrained on ImageNet!")
# ── Step 4: Feature Extraction ────────────
# Freeze ALL layers — don't train them
print("\n=== Approach 1: Feature Extraction ===")
for param in model.parameters():
    param.requires_grad = False
# Replace last layer for 10 classes
num_features   = model.fc.in_features
model.fc       = nn.Linear(num_features, 10)
print(f"Replaced last layer: {num_features} → 10")

# Only last layer trains
trainable = sum(p.numel() for p in model.parameters()
                if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params : {trainable:,}")
print(f"Total params     : {total:,}")
print(f"Frozen params    : {total-trainable:,}")
# ── Step 5: Train Feature Extraction ──────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0
    for images, labels in loader:
        outputs = model(images)
        loss    = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss/len(loader), correct/total
def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs      = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
print("\nTraining Feature Extraction (3 epochs)...")
fe_losses = []
fe_accs   = []
for epoch in range(3):
    loss, acc = train_epoch(model, train_loader,
                            criterion, optimizer)
    test_acc  = evaluate(model, test_loader)
    fe_losses.append(loss)
    fe_accs.append(test_acc)
    print(f"Epoch {epoch+1}/3 | "
          f"Loss: {loss:.4f} | "
          f"Train: {acc*100:.1f}% | "
          f"Test: {test_acc*100:.1f}%")
# ── Step 6: Fine Tuning ───────────────────
print("\n=== Approach 2: Fine Tuning ===")
model_ft = models.resnet18(pretrained=True)
# Unfreeze LAST block + final layer
for param in model_ft.parameters():
    param.requires_grad = False
# Unfreeze layer4 and fc
for param in model_ft.layer4.parameters():
    param.requires_grad = True
model_ft.fc = nn.Linear(model_ft.fc.in_features, 10)
trainable = sum(p.numel() for p in model_ft.parameters()
                if p.requires_grad)
print(f"Trainable params for fine-tuning: {trainable:,}")
optimizer_ft = optim.Adam(
    filter(lambda p: p.requires_grad,
           model_ft.parameters()),
    lr=0.0001   # smaller lr for fine tuning
)
print("\nTraining Fine Tuning (3 epochs)...")
ft_losses = []
ft_accs   = []
for epoch in range(3):
    loss, acc = train_epoch(model_ft, train_loader,
                            criterion, optimizer_ft)
    test_acc  = evaluate(model_ft, test_loader)
    ft_losses.append(loss)
    ft_accs.append(test_acc)
    print(f"Epoch {epoch+1}/3 | "
          f"Loss: {loss:.4f} | "
          f"Train: {acc*100:.1f}% | "
          f"Test: {test_acc*100:.1f}%")
# ── Step 7: Compare Results ───────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(fe_losses,  label='Feature Extraction', color='blue')
plt.plot(ft_losses,  label='Fine Tuning',        color='red')
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot([a*100 for a in fe_accs],
         label='Feature Extraction', color='blue')
plt.plot([a*100 for a in ft_accs],
         label='Fine Tuning',        color='red')
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.legend()
plt.suptitle("Transfer Learning Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("transfer_learning.png")
plt.show()
# ── Step 8: Final Summary ─────────────────
print("\n=== Final Summary ===")
print(f"Feature Extraction Test Acc : {fe_accs[-1]*100:.1f}%")
print(f"Fine Tuning Test Acc        : {ft_accs[-1]*100:.1f}%")
print("\nKey insight:")
print("Fine Tuning usually beats Feature Extraction!")
print("But needs smaller learning rate to avoid forgetting!")




