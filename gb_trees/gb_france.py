import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    dataPath = os.path.join(os.path.dirname(__file__), '../data')

    xFrance = pd.read_csv(os.path.join(dataPath, 'xFrance.csv')).to_numpy()
    yFrance = pd.read_csv(os.path.join(dataPath, 'yFrance.csv')).to_numpy().ravel()
    xTrain, xTest, yTrain, yTest = train_test_split(xFrance, yFrance, test_size=0.2, random_state=4)

    xEngland = pd.read_csv(os.path.join(dataPath, 'xEngland.csv')).to_numpy()
    yEngland = pd.read_csv(os.path.join(dataPath, 'yEngland.csv')).to_numpy().ravel()

    xGermany = pd.read_csv(os.path.join(dataPath, 'xGermany.csv')).to_numpy()
    yGermany = pd.read_csv(os.path.join(dataPath, 'yGermany.csv')).to_numpy().ravel()

    xItaly = pd.read_csv(os.path.join(dataPath, 'xItaly.csv')).to_numpy()
    yItaly = pd.read_csv(os.path.join(dataPath, 'yItaly.csv')).to_numpy().ravel()

    xSpain = pd.read_csv(os.path.join(dataPath, 'xSpain.csv')).to_numpy()
    ySpain = pd.read_csv(os.path.join(dataPath, 'ySpain.csv')).to_numpy().ravel()

    model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=100, learning_rate=0.01)
    model.fit(xTrain, yTrain)

    frHat = model.predict(xTest)
    frAcc = accuracy_score(yTest, frHat)
    frF1 = f1_score(yTest, frHat, average='weighted')
    print("Accuracy on Ligue 1:", frAcc)
    print("F1 Score on Ligue 1:", frF1)

    enHat = model.predict(xEngland)
    enAcc = accuracy_score(yEngland, enHat)
    enF1 = f1_score(yEngland, enHat, average='weighted')
    print("Accuracy on Premier League:", enAcc)
    print("F1 Score on Premier League:", enF1)

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

    spHat = model.predict(xSpain)
    spAcc = accuracy_score(ySpain, spHat)
    spF1 = f1_score(ySpain, spHat, average='weighted')
    print("Accuracy on La Liga:", spAcc)
    print("F1 Score on La Liga:", spF1)

    #ORIGINAL
    # Accuracy on Premier League: 0.38265550239234447
    # Accuracy on Bundesliga: 0.36822782929779513
    # Accuracy on Serie A: 0.4050308446829235
    # Accuracy on La Liga: 0.39949418137698867

    #CONDENSED TARGET
    # Accuracy on Ligue 1: 0.396747822996268
    # F1 Score on Ligue 1: 0.3654492030893935
    # Accuracy on Premier League: 0.3810321257689679
    # F1 Score on Premier League: 0.3503488713408159
    # Accuracy on Bundesliga: 0.36782462916198033
    # F1 Score on Bundesliga: 0.33692411400503647
    # Accuracy on Serie A: 0.40268972470479675
    # F1 Score on Serie A: 0.3800568749686776
    # Accuracy on La Liga: 0.39541003776551203
    # F1 Score on La Liga: 0.3657750201851831

    #FULL TARGET
    # Accuracy on Ligue 1: 0.3309934245601564
    # F1 Score on Ligue 1: 0.27305975292260715
    # Accuracy on Premier League: 0.3206083390293917
    # F1 Score on Premier League: 0.26294568501625754
    # Accuracy on Bundesliga: 0.31173736816416614
    # F1 Score on Bundesliga: 0.2583444892025933
    # Accuracy on Serie A: 0.3643261163041064
    # F1 Score on Serie A: 0.3182866995027068
    # Accuracy on La Liga: 0.3261846579743331
    # F1 Score on La Liga: 0.26819885446698444


if __name__=="__main__":
    main()