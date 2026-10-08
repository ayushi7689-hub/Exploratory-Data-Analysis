import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())
print("shape of dataset:", df.shape)
print(df.columns)
print(df.info())
print(df.describe())

print("Age mean:", df["Age"].mean())
print("Age median:", df["Age"].median())
print("Age Standard Deviation:", df["Age"].std())

print("Fare mean:", df["Fare"].mean())
print("Fare median:", df["Fare"].median())
print("Fare Standard Deviation:", df["Fare"].std())

#Age Histogram
plt.figure(figsize=(8,5))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Distribution of passanger Age")
plt.xlabel("Age")
plt.ylabel("Number of passangers")

plt.show()

#Fare Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["Fare"].dropna(), bins=30)
plt.title("Distribution of passanger Fare")
plt.xlabel("Fare")
plt.ylabel("Number of passangers")

plt.show()

#Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Fare"])
plt.title("Boxplot of passanger Fare")
plt.xlabel("Fare")
plt.show()

sns.pairplot(df[["Age", "Fare", "SibSp", "Parch", "Survived"]])
plt.show()

corr = df[["Age", "Fare", "SibSp", "Parch", "Survived"]].corr()

print(corr)

sns.heatmap(corr, annot=True)

plt.title("Correlation Matrix")
plt.show()