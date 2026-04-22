import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

df['Age']=df['Age'].fillna(df['Age'].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])         # male=1, female=0
df["Embarked"] = le.fit_transform(df["Embarked"])  # C=0, Q=1, S=2

# ── 3. Split Features and Target ─────────────────────────
X = df.drop(columns=["Survived"])
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importances:")
print(importance.to_string(index=False))

