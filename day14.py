import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,
                             classification_report)
from xgboost import XGBClassifier

# Load Data
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Fare']=df['Fare'].fillna(df['Fare'].mean())
df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked'] = df['Embarked'].fillna('S')
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[['Age', 'Sex', 'Pclass', 'Fare',
        'Embarked', 'SibSp', 'Parch']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("="*40)
print(" Week 2 Final Project ")
print("="*40)
print(f"Train : {len(X_train)} rows")
print(f"Test  : {len(X_test)} rows")
print(f"Features : {X.columns.tolist()}")

# ── Step 2: Train Classical ML Models ─────
models = {
    'Logistic Regression' : LogisticRegression(
                                max_iter=200),
    'Decision Tree'       : DecisionTreeClassifier(
                                max_depth=4,
                                random_state=42),
    'Random Forest'       : RandomForestClassifier(
                                n_estimators=100,
                                random_state=42),
    'XGBoost'             : XGBClassifier(
                                n_estimators=100,
                                random_state=42,
                                eval_metric='logloss'),
}

results = {}
print("\n=== Classical ML Models ===")
print(f"{'Model':<22} {'CV Acc':>8} {'CV Std':>8} "
      f"{'Test Acc':>10}")
print("-" * 52)
for name, model in models.items():
    # Cross validation
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='accuracy')

    # Train and evaluate
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    results[name] = {
        'cv_mean' : cv_scores.mean(),
        'cv_std'  : cv_scores.std(),
        'test_acc': test_acc
    }

    print(f"{name:<22} "
          f"{cv_scores.mean()*100:>7.1f}% "
          f"{cv_scores.std()*100:>7.1f}% "
          f"{test_acc*100:>9.1f}%")
# ── Step 3: Neural Network ────────────────
print("\n=== Neural Network ===")

# Scale data for neural network
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_tr_sc, X_te_sc, y_tr, y_te = train_test_split(
    X_sc, y.values, test_size=0.2, random_state=42)

X_tr_t = torch.FloatTensor(X_tr_sc)
X_te_t = torch.FloatTensor(X_te_sc)
y_tr_t = torch.FloatTensor(y_tr)
y_te_t = torch.FloatTensor(y_te)

# Build network
class SurvivalNet(nn.Module):
    def __init__(self):
        super(SurvivalNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
     return self.net(x)
nn_model  = SurvivalNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
# Train
train_losses = []
print("Training Neural Network...")
for epoch in range(150):
    nn_model.train()
    pred = nn_model(X_tr_t).squeeze()
    loss = criterion(pred, y_tr_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/150 | "
              f"Loss: {loss.item():.4f}")
# Evaluate neural network
nn_model.eval()
with torch.no_grad():
    nn_pred   = nn_model(X_te_t).squeeze()
    nn_labels = (nn_pred >= 0.5).float()
    nn_acc    = accuracy_score(
        y_te, nn_labels.numpy())
results['Neural Network'] = {
    'cv_mean' : nn_acc,
    'cv_std'  : 0,
    'test_acc': nn_acc
}
print(f"\nNeural Network Test Accuracy: {nn_acc*100:.1f}%")
# Best classical model report
best_classical = max(
    ['Logistic Regression', 'Decision Tree',
     'Random Forest', 'XGBoost'],
    key=lambda x: results[x]['test_acc']
)
best_model = models[best_classical]
print(f"\nBest Classical Model: {best_classical}")
print(classification_report(
    y_test,
    best_model.predict(X_test),
    target_names=['Not Survived', 'Survived']
))
# ── Step 5: Visualizations ────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Chart 1 — Test Accuracy Bar Chart
names    = list(results.keys())
test_acc = [results[n]['test_acc']*100 for n in names]
colors   = ['#6366F1', '#10B981', '#F59E0B',
            '#EC4899', '#8B5CF6']

bars = axes[0, 0].bar(names, test_acc, color=colors)
axes[0, 0].set_title("Test Accuracy — All Models",
                      fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel("Accuracy %")
axes[0, 0].set_ylim([70, 100])
axes[0, 0].tick_params(axis='x', rotation=20)
for bar, acc in zip(bars, test_acc):
    axes[0, 0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.3,
        f'{acc:.1f}%', ha='center', fontsize=9)
# Chart 2 — CV Accuracy with Error Bars
cv_means = [results[n]['cv_mean']*100
            for n in names[:4]]
cv_stds  = [results[n]['cv_std']*100
            for n in names[:4]]
axes[0, 1].bar(names[:4], cv_means,
               yerr=cv_stds, color=colors[:4],
               capsize=5)
axes[0, 1].set_title("CV Accuracy with Std Dev",
                      fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel("CV Accuracy %")
axes[0, 1].set_ylim([70, 100])
axes[0, 1].tick_params(axis='x', rotation=20)

# Chart 3 — Neural Network Training Loss
axes[1, 0].plot(train_losses, color='purple')
axes[1, 0].set_title("Neural Network Training Loss",
                      fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
# Chart 4 — Feature Importance from XGBoost
xgb_model  = models['XGBoost']
feat_imp   = pd.Series(
    xgb_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

feat_imp.plot(kind='barh', ax=axes[1, 1],
              color='steelblue')
axes[1, 1].set_title("XGBoost Feature Importance",
                      fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Importance Score")

plt.suptitle(
    "Week 2 Final Project — All Models Compared",
    fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
# ── Step 6: Final Summary ─────────────────
print("\n" + "=" * 45)
print("   WEEK 2 FINAL SUMMARY")
print("=" * 45)

for name in results:
    acc = results[name]['test_acc']
    bar = "█" * int(acc * 50)
    print(f"{name:<22} {bar} {acc*100:.1f}%")

best_overall = max(results,
                   key=lambda x: results[x]['test_acc'])
print(f"\n🏆 Best Model    : {best_overall}")
print(f"🎯 Best Accuracy : "
      f"{results[best_overall]['test_acc']*100:.1f}%")

print("\n✅ Week 2 Complete!")
print("📚 Models learned: Linear Regression, "
      "Logistic Regression,")
print("   Decision Tree, Random Forest, "
      "XGBoost, Neural Network")
print("🚀 Next: Week 3 — Deep Learning + "
      "CNN + Transfer Learning!")
