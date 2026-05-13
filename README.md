# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the California housing dataset and extract required input and output columns.
2. Split the dataset into training and testing sets using an 80-20 ratio.
3. Standardize the input and output features using `StandardScaler`.
4. Initialize `SGDRegressor` for linear regression with stochastic gradient descent.
5. Wrap the regressor with `MultiOutputRegressor` to handle multiple target variables.
6. Train the model on the scaled training data.
7. Predict on the test data and inverse-transform the predictions to original scale.
8. Compute and print the mean squared error between actual and predicted values.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: RIZWAN B
RegisterNumber: 212224100051
*/
```
```

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the California Housing dataset
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
# Use the first 3 features as inputs
X = df.drop(columns=['AveOccup','HousingPrice'])#data[:, :3]  # Features: 'MedInc', 'HouseAge', 'AveRooms'
# Use 'MedHouseVal' and 'AveOccup' as output variables
Y = df[['AveOccup','HousingPrice']]#np.column_stack((data.target, data.data[:, 6]))  # Targets: 'MedHouseVal', 'AveOccup'

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

# Optionally, print some predictions
print("\nPredictions:\n", Y_pred[:5])  # Print first 5 predictions
```

## Output:


<img width="726" height="470" alt="image" src="https://github.com/user-attachments/assets/e8e80b91-3ef2-4249-a15c-ec6b64d77f82" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
