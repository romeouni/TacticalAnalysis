from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

X = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis-main/data/xWhole.csv").to_numpy()
y = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis-main/data/yWhole.csv").to_numpy()

subs = np.where(y==17)[0]

X = np.delete(X, subs, axis=0)
y = np.delete(y, subs, axis=0)

n, d = X.shape
permutation = np.random.permutation(n)

X = X[permutation]
y = y[permutation].ravel()


xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=4)

# Scale the data
scaler = StandardScaler()
xTrain_scaled = scaler.fit_transform(xTrain)
xTest_scaled = scaler.transform(xTest)

# Define the logistic regression model
logreg = LogisticRegression(multi_class='multinomial', solver='sag')

# Define hyperparameters and their possible values
param_grid = {
    'C': [100],  # Regularization parameter
    'max_iter': [100, 500, 1000, 5000, 10000],  # Maximum number of iterations
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(xTrain_scaled, yTrain)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_logreg = LogisticRegression(multi_class='multinomial', solver='sag', **best_params)
best_logreg.fit(xTrain_scaled, yTrain)

# Make predictions on the test set
yTest_pred = best_logreg.predict(xTest_scaled)

# Evaluate accuracy on the test set
accuracy = accuracy_score(yTest, yTest_pred)
print("Test Accuracy:", accuracy)

# Best hyperparameters C: 100, max_iter: 500