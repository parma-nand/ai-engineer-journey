import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Bar Chart

# x=np.array([1,2,3,4])
# y=np.array([2,7,1,9])
# plt.plot(x, y)
# plt.title("Simple Line Plot")
# plt.xlabel("X Values")
# plt.ylabel("Y Values")

# plt.show()

# x=np.array(["Math","Physics","Chemistry"])
# y=np.array([89,45,60])
# plt.bar(x,y)
# plt.xlabel('Subjects')
# plt.ylabel('Marks')
# plt.show()

# url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
# df=pd.read_csv(url)
# plt.figure(figsize=(8,4))
# x=df['Survived'].sum()
# y=len(df['Survived'])-x
# plt.bar(['Survived','Not Survived'],[x,y],color=['red','green'])
# # df['Survived'].value_counts().plot(kind='bar',color=['red','green'])
# plt.title("Survived vs Not Survivived")
# plt.xlabel("0=No,1=Yes")
# plt.ylabel("Count")
# plt.savefig("bar_chart.png")
# plt.show()

# Histogram Data
# url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
# df=pd.read_csv(url)
# plt.figure(figsize=(8,4))
# df['Age'].plot(kind='hist', color='red')
# # plt.hist(df['Age'].dropna(),bins=10)
# plt.show()


# url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
# df=pd.read_csv(url)
# count_age=df['Age'].value_counts().sort_index()
# print(df['Age'].value_counts().sort_index().index)
# plt.figure(figsize=(8,4))
# plt.scatter(count_age.index,count_age.values)
# plt.show()
# x = np.linspace(0,10,100)
# y = np.sin(x)
# y_cos = np.cos(x)

# plt.plot(x,y,label="Sin X",color='steelblue',linestyle='--')
# plt.plot(x,y_cos,label="Cos X",color='coral',linestyle='--')
# plt.axhline(0, color='black', linewidth=0.8)   # X-axis line
# plt.axvline(0, color='black', linewidth=0.8)   # Y-axis line
# plt.xlabel("X Axis")
# plt.ylabel("Yaxis")
# plt.title("Graph BW X and Y")
# plt.legend(loc='upper right')
# plt.grid(True, linestyle='--', alpha=0.5,color='gray', linewidth=0.8)
# plt.show()

x = np.linspace(0, 10, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#                        ^  ^
#                    1 row  2 cols   → axes is a list of 2

axes[0].plot(x, np.sin(x), color='steelblue')
axes[0].set_title('Sin X')
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(x, np.cos(x), color='coral')
axes[1].set_title('Cos X')
axes[1].grid(True, linestyle='--', alpha=0.5)

fig.suptitle('1×2 Subplots', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()