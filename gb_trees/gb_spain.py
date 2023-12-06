import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    dataPath = os.path.join(os.path.dirname(__file__), '../data')

    xSpain = pd.read_csv(os.path.join(dataPath, 'xSpain.csv')).to_numpy()
    ySpain = pd.read_csv(os.path.join(dataPath, 'ySpain.csv')).to_numpy().ravel()
    xTrain, xTest, yTrain, yTest = train_test_split(xSpain, ySpain, test_size=0.2, random_state=4)

    xItaly = pd.read_csv(os.path.join(dataPath, 'xItaly.csv')).to_numpy()
    yItaly = pd.read_csv(os.path.join(dataPath, 'yItaly.csv')).to_numpy().ravel()

    xGermany = pd.read_csv(os.path.join(dataPath, 'xGermany.csv')).to_numpy()
    yGermany = pd.read_csv(os.path.join(dataPath, 'yGermany.csv')).to_numpy().ravel()

    xEngland = pd.read_csv(os.path.join(dataPath, 'xEngland.csv')).to_numpy()
    yEngland = pd.read_csv(os.path.join(dataPath, 'yEngland.csv')).to_numpy().ravel()

    xFrance = pd.read_csv(os.path.join(dataPath, 'xFrance.csv')).to_numpy()
    yFrance = pd.read_csv(os.path.join(dataPath, 'yFrance.csv')).to_numpy().ravel()

    model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=100, learning_rate=0.01)
    model.fit(xTrain, yTrain)

    spHat = model.predict(xTest)
    spAcc = accuracy_score(yTest, spHat)
    spF1 = f1_score(yTest, spHat, average='weighted')
    print("Accuracy on La Liga:", spAcc)
    print("F1 Score on La Liga:", spF1)
    
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

    gerHat = model.predict(xGermany)
    gerAcc = accuracy_score(yGermany, gerHat)
    gerF1 = f1_score(yGermany, gerHat, average='weighted')
    print("Accuracy on Bundesliga:", gerAcc)
    print("F1 Score on Bundesliga:", gerF1)

    itHat = model.predict(xItaly)
    itAcc = accuracy_score(yItaly, itHat)
    itF1 = f1_score(yItaly, itHat, average='weighted')
    print("Accuracy on Serie A:", itAcc)
    print("F1 Score on Serie A:", itF1)

    #ORIGINAL
    # Accuracy on Premier League: 0.3772043745727956
    # Accuracy on Ligue 1: 0.3775258135029945
    # Accuracy on Bundesliga: 0.3631772170702205
    # Accuracy on La Liga: 0.3894974281857175

    #CONDENSED TARGET
    # Accuracy on La Liga: 0.4645841521557101
    # F1 Score on La Liga: 0.432734032406282
    # Accuracy on Premier League: 0.3809637730690362
    # F1 Score on Premier League: 0.3378264282649395
    # Accuracy on Ligue 1: 0.38984165348593364
    # F1 Score on Ligue 1: 0.34741846375528374
    # Accuracy on Bundesliga: 0.3685249241347113
    # F1 Score on Bundesliga: 0.325967940226653
    # Accuracy on Serie A: 0.40966181923819617
    # F1 Score on Serie A: 0.3767209825206893

    #FULL TARGET
    # Accuracy on La Liga: 0.3233082706766917
    # F1 Score on La Liga: 0.2626911137079997
    # Accuracy on Premier League: 0.3172761449077239
    # F1 Score on Premier League: 0.25526380407229704
    # Accuracy on Ligue 1: 0.3250990776448844
    # F1 Score on Ligue 1: 0.26243898540114946
    # Accuracy on Bundesliga: 0.31477197971266685
    # F1 Score on Bundesliga: 0.25545435515840637
    # Accuracy on Serie A: 0.3621217040619286
    # F1 Score on Serie A: 0.3110501820893058

if __name__=="__main__":
    main()