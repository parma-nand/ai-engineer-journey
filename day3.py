import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# print("Shape:\n", df.shape)
# print("Columns\n",df.columns.to_list())
# print("Datatype \n",df.dtypes)
# print("Missinng values : \n",df.isnull().sum())
# print("Missinng values : \n",df.isna().sum()) #isnull() is same as isna()

# survivors=df[df['Survived']==1]
# print("\nSurvivors",len(survivors))
# print("Length of Survived: ",(df['Survived'] == 1).sum())

# print("\nAge & Fare\n",df[['Age','Fare']].head())
# print("Missing Age \n",df['Age'].isna().sum())
# df['Age']=df['Age'].fillna(df['Age'].mean())
# print("Missing Age Now\n",df['Age'].isna().sum())
print("Group By Pclass and Mean of Survived\n",df.groupby('Pclass')['Survived'].mean()*100)
