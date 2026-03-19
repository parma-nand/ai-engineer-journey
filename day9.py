import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,roc_auc_score,classification_report,roc_curve,confusion_matrix)

# Load data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

# Clean Data

print(df.shape)
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex']=df['Sex'].map({'male':1,'female':0})
df['Embarked']=df['Embarked'].map({'S':0,'C':1,'Q':2})

# Feature and label
x=df[['Age','Pclass','Fare','Embarked']]
y=df['Survived']
print("Features shape:", x.shape)
print("Label distribution:\n", y.value_counts())

# Split Data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"Train data lenght {len(x_train)}")
print(f"Train data lenght {len(x_test)}")

# Train Model
model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
print("Model trained Successfully")

# Predict

# ── Predict ───────────────────────────────
y_pred      = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)[:, 1]

# _____Evaluate_________________
accuracy=accuracy_score(y_test,y_pred)
auc=roc_auc_score(y_test,y_pred_prob)
print(f"Model Accuracy : {accuracy*100:0.2f}")
print(f"AUC-ROC : {auc*100:0.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Confusion Matrix ──────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ROC Curve ___________
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()








