from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/xWhole.csv").to_numpy()
y = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/yWhole.csv").to_numpy()

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

# Create a KNN classifier
knn = KNeighborsClassifier()

# Define the hyperparameters and their possible values
param_grid = {
    'n_neighbors': list(range(90, 116)),  # Try different values for the number of neighbors
    'weights': ['uniform'],  # Best weight: uniform
    'p': [2]  # Best p: 2
}

# Create a GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the model to the data
grid_search.fit(xTrain_scaled, yTrain)
# Get the results of the grid search
results = pd.DataFrame(grid_search.cv_results_)

# Extract the relevant information
k_values = results['param_n_neighbors'].astype(int)
accuracy_values = results['mean_test_score']

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_values, marker='o', linestyle='-', color='b')
plt.title('Accuracy at Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model
best_knn = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_knn.predict(xTest_scaled)
test_accuracy = accuracy_score(yTest, y_pred)
print("Test Accuracy:", test_accuracy)

# best n: 30, p:2, uniform