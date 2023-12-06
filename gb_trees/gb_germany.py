import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    dataPath = os.path.join(os.path.dirname(__file__), '../data')

    xGermany = pd.read_csv(os.path.join(dataPath, 'xGermany.csv')).to_numpy()
    yGermany = pd.read_csv(os.path.join(dataPath, 'yGermany.csv')).to_numpy().ravel()
    xTrain, xTest, yTrain, yTest = train_test_split(xGermany, yGermany, test_size=0.2, random_state=4)


    xEngland = pd.read_csv(os.path.join(dataPath, 'xEngland.csv')).to_numpy()
    yEngland = pd.read_csv(os.path.join(dataPath, 'yEngland.csv')).to_numpy().ravel()

    xFrance = pd.read_csv(os.path.join(dataPath, 'xFrance.csv')).to_numpy()
    yFrance = pd.read_csv(os.path.join(dataPath, 'yFrance.csv')).to_numpy().ravel()

    xItaly = pd.read_csv(os.path.join(dataPath, 'xItaly.csv')).to_numpy()
    yItaly = pd.read_csv(os.path.join(dataPath, 'yItaly.csv')).to_numpy().ravel()

    xSpain = pd.read_csv(os.path.join(dataPath, 'xSpain.csv')).to_numpy()
    ySpain = pd.read_csv(os.path.join(dataPath, 'ySpain.csv')).to_numpy().ravel()

    model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=100, learning_rate=0.01)
    model.fit(xTrain, yTrain)

    gerHat = model.predict(xTest)
    gerAcc = accuracy_score(yTest, gerHat)
    gerF1 = f1_score(yTest, gerHat, average='weighted')
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

    #ORIGINAL
    # Accuracy on Premier League: 0.37570061517429937
    # Accuracy on Ligue 1: 0.37430912225203933
    # Accuracy on Serie A: 0.3943847297458945
    # Accuracy on La Liga: 0.3849860728994002

    #CONDENSED TARGET
    # Accuracy on Bundesliga: 0.3563925729442971
    # F1 Score on Bundesliga: 0.3288647055456212
    # Accuracy on Premier League: 0.3710868079289132
    # F1 Score on Premier League: 0.34338790995232477
    # Accuracy on Ligue 1: 0.3748245037231868
    # F1 Score on Ligue 1: 0.3478810391617657
    # Accuracy on Serie A: 0.39028349766742426
    # F1 Score on Serie A: 0.3708486574113448
    # Accuracy on La Liga: 0.38056015994805104
    # F1 Score on La Liga: 0.35586361697681124

    #FULL TARGET
    # Accuracy on Bundesliga: 0.30981432360742706
    # F1 Score on Bundesliga: 0.2478677710301022
    # Accuracy on Premier League: 0.310850991114149
    # F1 Score on Premier League: 0.24633340897119613
    # Accuracy on Ligue 1: 0.3097264923847945
    # F1 Score on Ligue 1: 0.2460108418080554
    # Accuracy on Serie A: 0.3499547155624669
    # F1 Score on Serie A: 0.2976999294467179
    # Accuracy on La Liga: 0.31572651617423403
    # F1 Score on La Liga: 0.25257379370536687

if __name__=="__main__":
    main()