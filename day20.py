import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,roc_auc_score,root_mean_squared_error,mean_squared_error

# arr=np.array([[1,2,3],[4,5,6]])
# print(arr.mean())
# a=np.ones([2,2])
# print(a)
# d=np.arange(0,40,3)
# print(d)
# e=np.random.randint(0,100,(3,3))
# print(e)
# f = np.random.randn(0,100,(3, 3))
# print(f)
# arr = np.array([10, 20, 30, 40, 50, 60])
# print((arr[:4]))
# print(arr[::3])
# print(arr[::-1])
# arr=np.array([[1,2,3],
#             [4,5,6],
#             [7,8,9]])


# print(arr[0:2, 1:3])
# print(arr[:, 0:3])
# url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
# df=pd.read_csv(url)
# print(df.isnull().sum())

# df['Age']=df['Age'].fillna(df['Age'].mean())
# df=df.drop(columns=(['Cabin','Embarked']))
# print(df.isnull().sum())

# print(df.groupby('Survived')[['Age','Pclass']].mean().reset_index())
# corr = df.select_dtypes(include='number').corr()
# plt.figure(figsize=(8,6))
# sns.heatmap(corr,annot=True,cmap='coolwarm')
# plt.show()

# Linear Regression
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
X=df[['Age','Pclass']]
Y=df['Fare']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)
print(f"Coefficient   : {model.coef_}")
print(f"Intercept     : {model.intercept_:.2f}")

y_pred=model.predict(x_test)
print("\nFirst 5 predictions:", y_pred[:5].round(1))
print("First 5 actual     :", y_test.values[:5])

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"MSE        : {mse:.2f}")
print(f"RMSE       : {rmse:.2f}")
print(f"R2 Score   : {r2:.2f}")

