import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split,cross_val_score,KFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
from xgboost import XGBClassifier

# Load and clean data
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

df["Age"]=df['Age'].fillna(df['Age'].mean())
df['Fare']=df['Fare'].fillna(df['Fare'].mean())
df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked']=df['Embarked'].fillna('S')
df['Embarked']=df['Embarked'].map({'S':0,'Q':1,'C':2})

x=df[['Age','Fare','Pclass','Sex','SibSp','Parch']]
y=df['Survived']

print("Dataset Shape : ",df.shape)
print("Features used : ",x.columns.to_list())

# Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"Train data : {len(x_train)}")
print(f"Test data : {len(x_test)}")


# XGBoost Model

xgb=XGBClassifier(
    n_estimators=100,
    max_deepth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss'
    
)
xgb.fit(x_train,y_train)
xgb_pred=xgb.predict(x_test)
xgb_accuracy=accuracy_score(y_test,xgb_pred)
print(f"XGBoost Model Accuracy : {xgb_accuracy:0.2f}")
print("Classification Report :")
print(classification_report(y_test,xgb_pred))

# COnfusion Matrix
cm=confusion_matrix(y_test,xgb_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues', xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Features importance

feature_imp=pd.Series(xgb.feature_importances_,index=x.columns).sort_values(ascending=False)
print(feature_imp)
# plt.figure(figsize=(8,6))
feature_imp.plot(kind='bar')
plt.xlabel("Important Features")
plt.ylabel("Scores")
plt.tight_layout()
plt.show()

# K fold Cross validation
print("\n=== K-Fold Cross Validation ===")
kf=KFold(n_splits=5,shuffle=True,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
xgb2=XGBClassifier(n_estimators=100,random_state=42,eval_metric='logloss')

rf_scores=cross_val_score(rf,x,y,cv=kf,scoring='accuracy')
xgb2_scores=cross_val_score(xgb2,x,y,cv=kf,scoring='accuracy')
print("\nRandom Forest CV Scores :", rf_scores.round(3))
print(f"Mean : {rf_scores.mean()*100:.1f}%"
      f" | Std: {rf_scores.std()*100:.1f}%")

print("\nXGBoost CV Scores :", xgb2_scores.round(3))
print(f"Mean : {xgb2_scores.mean()*100:.1f}%"
      f" | Std: {xgb2_scores.std()*100:.1f}%")

print("\n=== Hyperparameter Experiment ===")
learning_rates = [0.001, 0.01, 0.1, 0.3, 0.5]
lr_scores      = []

for lr in learning_rates:
    model = XGBClassifier(
        n_estimators  = 100,
        learning_rate = lr,
        random_state  = 42,
        eval_metric   = 'logloss'
    )
    scores = cross_val_score(model, x, y, cv=3,
                             scoring='accuracy')
    lr_scores.append(scores.mean())
    print(f"LR={lr:.3f} → CV Accuracy: {scores.mean()*100:.1f}%")
    
    # ── Step 8: Learning Rate Comparison ──────
plt.figure(figsize=(8, 4))
plt.plot(learning_rates, lr_scores,
         marker='o', color='blue', linewidth=2)
plt.title("Learning Rate vs CV Accuracy")
plt.xlabel("Learning Rate")
plt.ylabel("CV Accuracy")
plt.xscale('log')
plt.savefig("learning_rate_experiment.png")
plt.show()

# ── Step 9: Final Comparison ──────────────
print("\n=== Final Comparison ===")
print(f"Random Forest CV : {rf_scores.mean()*100:.1f}%"
      f" (+/- {rf_scores.std()*100:.1f}%)")
print(f"XGBoost       CV : {xgb2_scores.mean()*100:.1f}%"
      f" (+/- {xgb2_scores.std()*100:.1f}%)")
print(f"\nBest Learning Rate: "
      f"{learning_rates[np.argmax(lr_scores)]}")