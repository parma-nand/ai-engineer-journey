import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

class TitanicAnalyzer:
    def __init__(self,df):
        self.df=df

    def clean(self):
        # self.df['Age'].fillna(self.df['Age'].mean(), inplace=True) it is also correct but not recomended
        self.df['Age']=self.df['Age'].fillna(self.df['Age'].mean())
        self.df['Embarked']=self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        self.df=self.df.drop(columns=['Cabin','Name','Ticket'])
        return self.df
    def analyze(self):
        print("="*35)
        print("Titaninc Data Analysis")
        print("="*35)
        print(f"Total Passenger {len(self.df)}")
        print(f"Total Passenger Survived {self.df['Survived'].sum()}")
        print(f"Survival Rate {self.df['Survived'].mean()*100:.1f}%")
        print(f" \n By Gender {self.df.groupby('Pclass')['Survived'].mean()}")
        print(f" \n By Gender {self.df.groupby('Sex')['Survived'].mean()}")
    def visualize(self):
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        self.df['Survived'].value_counts().plot(kind='bar',ax=axes[0],color=['red','green'])
        axes[0].set_title("Survived vs Not")

        sns.boxplot(x=self.df['Pclass'],y='Age',data=self.df,ax=axes[1]).set_title("Boxplot for Class and Age")
        
        sns.countplot(x='Sex', hue='Survived', data=self.df, ax=axes[2])
        axes[2].set_title("Survival by Gender")

        plt.tight_layout()
        plt.show()


analyzer=TitanicAnalyzer(df)
analyzer.clean()
analyzer.analyze()
analyzer.visualize()

