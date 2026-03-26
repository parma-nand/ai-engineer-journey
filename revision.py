import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)]
) 
train_dataset=torchvision.datasets.MNIST(
    root='./data1',
    train=True,
    download=True,
    transform=transform
    
)
test_dataset=torchvision.datasets.MNIST(
    root='./data1',
    train=False,
    download=True,
    transform=transform
    
)
train_loader=torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_loader=torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
# image,label = test_dataset[0]
# print(test_dataset[0][1])

# print(label)
# print("___UpToDate__")
# plt.imshow(image.squeeze(),cmap='gray')
# plt.title(f"Label: {label}")
# plt.show()

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer=nn.Sequential(
        nn.Conv2d(1,32,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32,64,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 10) 
        )
        # self.fc_layer=nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(64 * 7 * 7, 128),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(128, 10) 
        # )
    def forward(self, x):
        x = self.conv_layer(x)
        return x
model=ConvModel()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
train_losses = []
train_accs   = []
for epoch in range(5):
    model.train()
    
    for i,(images, labels) in enumerate(train_loader):
        output=model(images)
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()}")
        
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)

        # get predicted class
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")
    
    
        




