import numpy as np
import pandas as pd

url ="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

def load_data(url):
    return pd.read_csv(url)

# print(load_data(url).head())

def get_missing_values(df):
    return df.isnull().sum()[df.isnull().sum()>0]

# print(get_missing_values(df))

def filling_missing_values(df):
    df['Age']=df['Age'].fillna(df['Age'].mean())
    return df
# filling_missing_values(df)

# print(get_missing_values(df))

def get_srvival_rate_by_group(df,column):
    return df.groupby(column)['Survived'].mean()

# Calling Function

print("Missing values:\n", get_missing_values(df))
df = filling_missing_values(df)
print("Missing values:\n", get_missing_values(df))
print("\nSurvival by gender:\n", get_srvival_rate_by_group(df, 'Sex'))
print("\nSurvival by class:\n", get_srvival_rate_by_group(df, 'Pclass'))

# OOPs

class DataExplorer:
    def __init__(self,df):
        self.df=df
    def shape(self):
        return self.df.shape
    def missing(self):
        return self.df.isnull().sum()
    def summary(self):
        print("shape : ",self.shape())
        print("Missing : \n",self.missing())

# Use the class
explorer = DataExplorer(df)
explorer.summary()

