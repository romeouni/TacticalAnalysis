import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = pd.read_csv("data/xWhole.csv").to_numpy()
y = pd.read_csv("data/yWhole.csv").to_numpy()

subs = np.where(y==17)[0]

X = np.delete(X, subs, axis=0)
y = np.delete(y, subs, axis=0)

n, d = X.shape
permutation = np.random.permutation(n)

X = X[permutation]
y = y[permutation].ravel()


xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=4)

# Define the neural network architecture
def create_model(optimizer='adam', activation='relu', neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=xTrain.shape[1], activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create a KerasClassifier based on the model
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Define the hyperparameters to search
param_grid = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'activation': ['relu', 'sigmoid'],
    'neurons': [16, 32, 64]
}

# Use GridSearchCV to find the best hyperparameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(xTrain, yTrain)

# Print the best hyperparameters and corresponding accuracy
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Evaluate the model with the best hyperparameters on the test set
y_pred = grid_result.best_estimator_.predict(xTest)
accuracy = accuracy_score(yTest, y_pred)
print("Test Accuracy: %.2f%%" % (accuracy * 100))
