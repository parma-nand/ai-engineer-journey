import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# load data
url ="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)
print(df)

class LinearRegressionModel:
    def __init__(self,df):
        self.df=df
    
    # Prepare data or Clean Data
    def clean(self):
        self.df=self.df.dropna(subset=['Age','Fare'])

# Features = what we give the model (input) & Label = what we want to predict (output)
    def features(self): 
        self.X=self.df[['Age','Pclass']]
        self.Y=self.df['Fare']
        print("\nFeatures (X):")
        print(self.X.head())
        print("\nLabel (y):")
        print(self.Y.head())
# Split Data
    def split(self):
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.2,random_state=42)
        print(f"Traing Data {len(self.X_train)} rows")
        print(f"Testing Data {len(self.X_test)} rows")
        
        # Split Data
    def train(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train,self.Y_train)
        print("\nModel trained!")
        print(f"Coefficient   : {self.model.coef_}")
        print(f"Intercept     : {self.model.intercept_:.2f}")
    
    # Predict  
    def predict(self):
        self.y_pred=self.model.predict(self.X_test)
        print("\nFirst 5 predictions:", self.y_pred[:5].round(1))
        print("First 5 actual     :", self.Y_test.values[:5])
        
    def evaluate(self):
        # ── Step 6: Evaluate ──────────────────────
        mse = mean_squared_error(self.Y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.Y_test, self.y_pred)

        print("\n--- Model Performance ---")
        print(f"MSE        : {mse:.2f}")
        print(f"RMSE       : {rmse:.2f}")
        print(f"R2 Score   : {r2:.2f}")
    def visualize(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.Y_test, self.y_pred, alpha=0.5, color='blue', label='Predictions')
        plt.plot([self.Y_test.min(), self.Y_test.max()],
         [self.Y_test.min(), self.Y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel("Actual Age")
        plt.ylabel("Predicted Age")
        plt.title("Actual vs Predicted Age")
        plt.legend()
        plt.savefig("linear_regression.png")
        plt.show()
        print("\nChart saved!")


    
analyzer = LinearRegressionModel(df)
analyzer.clean()
analyzer.features()
analyzer.split()
analyzer.train()
analyzer.predict()
analyzer.evaluate()
analyzer.visualize()
