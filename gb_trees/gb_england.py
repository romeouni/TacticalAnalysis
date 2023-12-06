import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

def main():
    dataPath = os.path.join(os.path.dirname(__file__), '../data')

    xEngland = pd.read_csv(os.path.join(dataPath, 'xEngland.csv')).to_numpy()
    yEngland = pd.read_csv(os.path.join(dataPath, 'yEngland.csv')).to_numpy().ravel()
    xTrain, xTest, yTrain, yTest = train_test_split(xEngland, yEngland, test_size=0.2, random_state=4)

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

    enHat = model.predict(xTest)
    enAcc = accuracy_score(yTest, enHat)
    enF1 = f1_score(yTest, enHat, average='weighted')
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

    spHat = model.predict(xSpain)
    spAcc = accuracy_score(ySpain, spHat)
    spF1 = f1_score(ySpain, spHat, average='weighted')
    print("Accuracy on La Liga:", spAcc)
    print("F1 Score on La Liga:", spF1)


    #ORIGINAL
    # Accuracy on Ligue 1: 0.3883310526222254
    # Accuracy on Bundesliga: 0.368546145194491
    # Accuracy on Serie A: 0.4056972948956749
    # Accuracy on La Liga: 0.3940600488730156 

    #CONDENSED TARGET
    # Accuracy on Premier League: 0.38345864661654133
    # F1 Score on Premier League: 0.3532117163456673
    # Accuracy on Ligue 1: 0.3867493646590485
    # F1 Score on Ligue 1: 0.355822266060966
    # Accuracy on Bundesliga: 0.3648749018525985
    # F1 Score on Bundesliga: 0.3349862887116439
    # Accuracy on Serie A: 0.40453527914010834
    # F1 Score on Serie A: 0.3808584312323889
    # Accuracy on La Liga: 0.38983919752559
    # F1 Score on La Liga: 0.360769487189377

    #FULL TARGET
    # Accuracy on Premier League: 0.3179254955570745
    # F1 Score on Premier League: 0.2525691945778334
    # Accuracy on Ligue 1: 0.3238905969539178
    # F1 Score on Ligue 1: 0.25561447680840826
    # Accuracy on Bundesliga: 0.31362604248456166
    # F1 Score on Bundesliga: 0.25196243803251694
    # Accuracy on Serie A: 0.3671115364240674
    # F1 Score on Serie A: 0.3128108345289699
    # Accuracy on La Liga: 0.32456125360993865
    # F1 Score on La Liga: 0.2579153125394861

if __name__=="__main__":
    main()