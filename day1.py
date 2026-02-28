import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Explore data
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Plot
df['Age'].dropna().hist(color='blue', edgecolor='black')
plt.title("Titanic - Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()