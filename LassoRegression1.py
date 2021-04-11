import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset=pd.read_csv("Data/Real-Data/Real_Combine.csv")
dataset=dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.lasso_model import LassoRegression
lin_reg = LassoRegression()
lin_reg.fit(X_train, y_train)
y_pred=lin_reg.predict(X_test)

sns.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
import pickle
# open a file, where you ant to store the data
file = open('lasso_regression_model.pkl', 'wb')
# dump information to that file
pickle.dump(lin_reg, file)
