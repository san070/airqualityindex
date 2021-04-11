import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("Data/Real-Data/Real_Combine.csv")

#Seeing the information of the dataset
dataset.info()

#Checking if there are any null values
dataset.isnull().sum()

#Checking for null values using heatmap
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Deleting all the rows with null values
dataset=dataset.dropna()
sns.pairplot(dataset)

#Correlation map
plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr(),annot=True)

#Correlation with respect to Independent variable
dataset.corrwith(dataset["PM 2.5"]).plot.bar(figsize=(20,10),title="Correlation with independent variable",fontsize=15,rot=45,grid=True)
sns.distplot(dataset["PM 2.5"])