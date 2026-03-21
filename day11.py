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

