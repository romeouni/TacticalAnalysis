import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    dataPath = os.path.join(os.path.dirname(__file__), '../data')

    xGermany = pd.read_csv(os.path.join(dataPath, 'xGermany.csv')).to_numpy()
    yGermany = pd.read_csv(os.path.join(dataPath, 'yGermany.csv')).to_numpy().ravel()

    xEngland = pd.read_csv(os.path.join(dataPath, 'xEngland.csv')).to_numpy()
    yEngland = pd.read_csv(os.path.join(dataPath, 'yEngland.csv')).to_numpy().ravel()

    xFrance = pd.read_csv(os.path.join(dataPath, 'xFrance.csv')).to_numpy()
    yFrance = pd.read_csv(os.path.join(dataPath, 'yFrance.csv')).to_numpy().ravel()

    xItaly = pd.read_csv(os.path.join(dataPath, 'xItaly.csv')).to_numpy()
    yItaly = pd.read_csv(os.path.join(dataPath, 'yItaly.csv')).to_numpy().ravel()

    xSpain = pd.read_csv(os.path.join(dataPath, 'xSpain.csv')).to_numpy()
    ySpain = pd.read_csv(os.path.join(dataPath, 'ySpain.csv')).to_numpy().ravel()

    model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=100, learning_rate=0.01)
    model.fit(xGermany, yGermany)

    gerHat = model.predict(xGermany)
    gerAcc = accuracy_score(yGermany, gerHat)
    gerF1 = f1_score(yGermany, gerHat, average='weighted')
    print("Accuracy on Bundesliga:", gerAcc)
    print("F1 Score on Bundesliga:", gerF1)
   
    enHat = model.predict(xEngland)
    enAcc = accuracy_score(yEngland, enHat)
    enF1 = f1_score(yEngland, enHat, average='weighted')
    print("Accuracy on Premier League:", enAcc)
    print("F1 Score on Premier League:", enF1)

    frHat = model.predict(xFrance)
    frAcc = accuracy_score(yFrance, frHat)
    frF1 = f1_score(yFrance, frHat, average='weighted')
    print("Accuracy on Ligue 1:", frAcc)
    print("F1 Score on Ligue 1:", frF1)

    itHat = model.predict(xItaly)
    itAcc = accuracy_score(yItaly, itHat)
    itF1 = f1_score(yItaly, itHat, average='weighted')
    print("Accuracy on Serie A:", itAcc)
    print("F1 Score on Serie A:", itF1)

    spHat = model.predict(xSpain)
    spAcc = accuracy_score(ySpain, spHat)
    spF1 = f1_score(ySpain, spHat, average='weighted')
    print("Accuracy on La Liga:", spAcc)
    print("F1 Score on La Liga:", spF1)

    # Accuracy on Premier League: 0.37570061517429937
    # Accuracy on Ligue 1: 0.37430912225203933
    # Accuracy on Serie A: 0.3943847297458945
    # Accuracy on La Liga: 0.3849860728994002

    # Accuracy on Bundesliga: 0.47151072724571863
    # F1 Score on Bundesliga: 0.45394739772498194
    # Accuracy on Premier League: 0.37664046479835955
    # F1 Score on Premier League: 0.3482910974622872
    # Accuracy on Ligue 1: 0.3759618973146848
    # F1 Score on Ligue 1: 0.34771792634561755
    # Accuracy on Serie A: 0.39385498726909207
    # F1 Score on Serie A: 0.3728098982023989
    # Accuracy on La Liga: 0.3844563304225978
    # F1 Score on La Liga: 0.3586722681347026

if __name__=="__main__":
    main()