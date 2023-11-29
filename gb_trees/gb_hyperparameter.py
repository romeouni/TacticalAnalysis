import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

def main():
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
    gbc = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=100, learning_rate=0.01)

    # paramGrid = {
    #     "max_iter": [3, 10, 30, 100, 300, 1000],
    #     "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    #     "learning_rate": np.arange(0.01, 0.1, 0.05)
    #     # In this model, other hyperparameters are coupled to these three, so no need to optimize them.
    # }

    # gridSearch = GridSearchCV(estimator=gbc, param_grid=paramGrid, cv=5, scoring='accuracy')
    # gridResult = gridSearch.fit(xTrain, yTrain)

    # print("Best parameters:", gridResult.best_params_)

    # optimalModel = gridResult.best_estimator_
    # accuracy = optimalModel.score(xTest, yTest)
    # print("Accuracy of optimal model on test set:", accuracy)

    # Best parameters: {'learning_rate': 0.01, 'max_iter': 300, 'max_leaf_nodes': 100}
    # Accuracy of optimal model on test set: 0.4777218349859052   

    gbc.fit(xTrain, yTrain)
    yHat = gbc.predict(yTest)

    accuracy = accuracy_score(yTest, yHat)

    print(accuracy)


if __name__ == "__main__":
    main()