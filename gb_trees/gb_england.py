import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import os

def main():
    dataPath = os.path.join(os.path.dirname(__file__), '../data')

    xTrain = pd.read_csv(os.path.join(dataPath, 'xEngland.csv')).to_numpy()
    yTrain = pd.read_csv(os.path.join(dataPath, 'yEngland.csv')).to_numpy().ravel()

    xFrance = pd.read_csv(os.path.join(dataPath, 'xFrance.csv')).to_numpy()
    yFrance = pd.read_csv(os.path.join(dataPath, 'yFrance.csv')).to_numpy().ravel()

    xGermany = pd.read_csv(os.path.join(dataPath, 'xGermany.csv')).to_numpy()
    yGermany = pd.read_csv(os.path.join(dataPath, 'yGermany.csv')).to_numpy().ravel()

    xItaly = pd.read_csv(os.path.join(dataPath, 'xItaly.csv')).to_numpy()
    yItaly = pd.read_csv(os.path.join(dataPath, 'yItaly.csv')).to_numpy().ravel()

    xSpain = pd.read_csv(os.path.join(dataPath, 'xSpain.csv')).to_numpy()
    ySpain = pd.read_csv(os.path.join(dataPath, 'ySpain.csv')).to_numpy().ravel()

    model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=100, learning_rate=0.01)
    model.fit(xTrain, yTrain)

    frHat = model.predict(xFrance)
    frAcc = accuracy_score(yFrance, frHat)
    print("Accuracy on Ligue 1:", frAcc)

    gerHat = model.predict(xGermany)
    gerAcc = accuracy_score(yGermany, gerHat)
    print("Accuracy on Bundesliga:", gerAcc)

    itHat = model.predict(xItaly)
    itAcc = accuracy_score(yItaly, itHat)
    print("Accuracy on Serie A:", itAcc)

    spHat = model.predict(xSpain)
    spAcc = accuracy_score(ySpain, spHat)
    print("Accuracy on La Liga:", spAcc)

    # Accuracy on Ligue 1: 0.3883310526222254
    # Accuracy on Bundesliga: 0.368546145194491
    # Accuracy on Serie A: 0.4056972948956749
    # Accuracy on La Liga: 0.3940600488730156 

if __name__=="__main__":
    main()