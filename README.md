# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Dataset Exploration
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nSummary statistics:")
print(df.describe())

# Data Splitting
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

# Linear Regression (using a sample dataset)
# Assuming a dataset with 'YearsExperience' and 'Salary'
data = {'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]}
df_lr = pd.DataFrame(data)

X_lr = df_lr[['YearsExperience']]
y_lr = df_lr['Salary']

X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_lr_train, y_lr_train)

y_lr_pred = lr_model.predict(X_lr_test)

mse = mean_squared_error(y_lr_test, y_lr_pred)
print("\nLinear Regression MSE:", mse)
 Data-Science-Assignment
