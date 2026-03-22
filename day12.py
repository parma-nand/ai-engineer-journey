import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Load Data and Clean Data
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

df["Age"]=df['Age'].fillna(df['Age'].mean())
df['Fare']=df['Fare'].fillna(df['Fare'].mean())
df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked']=df['Embarked'].fillna('S')
df['Embarked']=df['Embarked'].map({'S':0,'Q':1,'C':2})

X = df[['Age', 'Sex', 'Pclass', 'Fare', 'Embarked']].values
y = df['Survived'].values


# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data into Tensor
X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test  = torch.FloatTensor(y_test)

# Build Neural Network
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()   # for binary classification
)

# Loss and Optimizer
criterion = nn.BCELoss()          # binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Loop 
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train).squeeze()  # ✅ squeeze added

    # Loss
    loss = criterion(y_pred, y_train)  # ✅ correct order

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Prediction
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5).float()

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
        
print("Hii")

