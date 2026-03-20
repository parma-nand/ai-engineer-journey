import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load Data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Hiiii")

# Clean Data
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"] = df["Embarked"].map({"S": 0, "Q": 1, "C": 2})

print(df.shape)
print(df["Survived"].value_counts())

# Features and Label
x = df[["Age", "Sex", "Pclass", "Fare", "Embarked", "SibSp", "Parch"]]
y = df["Survived"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(f"Train Size : {len(x_train)}")
print(f"Test Size : {len(x_test)}")

# Decision Tree

print("-----------Printig Decision tree--------")
# Shallow tree — underfitting
dt_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
dt_shallow.fit(x_train, y_train)
shallow_acc = accuracy_score(y_test, dt_shallow.predict(x_test))

# Medium tree — mediumfitting
dt_medium = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_medium.fit(x_train, y_train)
medium_acc = accuracy_score(y_test, dt_medium.predict(x_test))

# Deep tree — mediumfitting
dt_deep = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_deep.fit(x_train, y_train)
deep_acc = accuracy_score(y_test, dt_deep.predict(x_test))

print(f"Shalllow Accuracy : {shallow_acc:0.2f}")
print(f"Medium Accuracy : {medium_acc:0.2f}")
print(f"Deep Accuracy : {deep_acc:0.2f}")
print("Deep tree is not always good")

# Visualize Decision Tree

plt.figure(figsize=(20, 8))
plot_tree(
    dt_medium,
    feature_names=x.columns.tolist(),
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree (depth=4)")
plt.show()

# Random Forest

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42  # 100 trees  # each tree max depth 6
)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy : {rf_accuracy : 0.2f}")

print("Classification Report :  ")
print(classification_report(y_test, rf_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"],
)
plt.title("Random Forest Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")


# ── Step 7: Feature Importance ────────────
feat_imp = pd.Series(rf_model.feature_importances_, index=x.columns).sort_values(
    ascending=False
)
plt.figure(figsize=(8, 4))
feat_imp.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Feature Importance — Random Forest")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.tight_layout()
plt.show()
print("\nFeature Importance:")
print(feat_imp)

# ───────── Final Model Comparison ────────
print("\n=== Final Comparison ===")
print(f"Decision Tree Shallow : {shallow_acc*100:.1f}%")
print(f"Decision Tree Medium  : {medium_acc*100:.1f}%")
print(f"Decision Tree Deep    : {deep_acc*100:.1f}%")
print(f"Random Forest         : {rf_accuracy*100:.1f}%")
