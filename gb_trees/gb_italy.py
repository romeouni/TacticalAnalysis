import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    dataPath = os.path.join(os.path.dirname(__file__), '../data')

    xItaly = pd.read_csv(os.path.join(dataPath, 'xItaly.csv')).to_numpy()
    yItaly = pd.read_csv(os.path.join(dataPath, 'yItaly.csv')).to_numpy().ravel()
    xTrain, xTest, yTrain, yTest = train_test_split(xItaly, yItaly, test_size=0.2, random_state=4)


    xGermany = pd.read_csv(os.path.join(dataPath, 'xGermany.csv')).to_numpy()
    yGermany = pd.read_csv(os.path.join(dataPath, 'yGermany.csv')).to_numpy().ravel()

    xEngland = pd.read_csv(os.path.join(dataPath, 'xEngland.csv')).to_numpy()
    yEngland = pd.read_csv(os.path.join(dataPath, 'yEngland.csv')).to_numpy().ravel()

    xFrance = pd.read_csv(os.path.join(dataPath, 'xFrance.csv')).to_numpy()
    yFrance = pd.read_csv(os.path.join(dataPath, 'yFrance.csv')).to_numpy().ravel()

    xSpain = pd.read_csv(os.path.join(dataPath, 'xSpain.csv')).to_numpy()
    ySpain = pd.read_csv(os.path.join(dataPath, 'ySpain.csv')).to_numpy().ravel()

    model = HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=100, learning_rate=0.01)
    model.fit(xTrain, yTrain)

    itHat = model.predict(xTest)
    itAcc = accuracy_score(yTest, itHat)
    itF1 = f1_score(yTest, itHat, average='weighted')
    print("Accuracy on Serie A:", itAcc)
    print("F1 Score on Serie A:", itF1)
    
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

    spHat = model.predict(xSpain)
    spAcc = accuracy_score(ySpain, spHat)
    spF1 = f1_score(ySpain, spHat, average='weighted')
    print("Accuracy on La Liga:", spAcc)
    print("F1 Score on La Liga:", spF1)

    #CONDENSED TARGET
    # Accuracy on Serie A: 0.4182330827067669
    # F1 Score on Serie A: 0.3671744494432291
    # Accuracy on Premier League: 0.3740088858509911
    # F1 Score on Premier League: 0.3132032873928519
    # Accuracy on Ligue 1: 0.3727096625139953
    # F1 Score on Ligue 1: 0.31065918006110504
    # Accuracy on Bundesliga: 0.36084290049445067
    # F1 Score on Bundesliga: 0.30225954789660586
    # Accuracy on La Liga: 0.3850544267673747
    # F1 Score on La Liga: 0.3250427208438268

    #FULL TARGET
    # Accuracy on Serie A: 0.39166097060833904
    # F1 Score on Serie A: 0.31816894702001813
    # Accuracy on Premier League: 0.3226930963773069
    # F1 Score on Premier League: 0.24160809893506816
    # Accuracy on Ligue 1: 0.3239261405036521
    # F1 Score on Ligue 1: 0.24139875542792918
    # Accuracy on Bundesliga: 0.3146446533539885
    # F1 Score on Bundesliga: 0.24060761368398526
    # Accuracy on La Liga: 0.3251080845537347
    # F1 Score on La Liga: 0.24552362141202203


if __name__=="__main__":
    main()