# C: 100, max_iter: 500
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get datasets (5)
xEngland = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/xEngland.csv").to_numpy()
yEngland = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/yEngland.csv").to_numpy()

xSpain = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/xSpain.csv").to_numpy()
ySpain = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/ySpain.csv").to_numpy()

xItaly = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/xItaly.csv").to_numpy()
yItaly = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/yItaly.csv").to_numpy()

xFrance = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/xFrance.csv").to_numpy()
yFrance = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/yFrance.csv").to_numpy()

xGermany = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/xGermany.csv").to_numpy()
yGermany = pd.read_csv("/Users/dylanethan/Desktop/TacticalAnalysis/data/yGermany.csv").to_numpy()

# Helper function to get scaled train data and test data
def trainTest(xData, yData):
    n = xData.shape[0]
    permutation = np.random.permutation(n)

    xData = xData[permutation]
    yData = yData[permutation].ravel()

    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=4)

    return xTrain, xTest, yTrain, yTest

# Get different train test data
xTrain_E_scaled, xTest_E_scaled, yTrain_E, yTest_E = trainTest(xEngland, yEngland)
xTrain_S_scaled, xTest_S_scaled, yTrain_S, yTest_S = trainTest(xSpain, ySpain)
xTrain_I_scaled, xTest_I_scaled, yTrain_I, yTest_I = trainTest(xItaly, yItaly)
xTrain_F_scaled, xTest_F_scaled, yTrain_F, yTest_F = trainTest(xFrance, yFrance)
xTrain_G_scaled, xTest_G_scaled, yTrain_G, yTest_G = trainTest(xGermany, yGermany)

# Define the logistic regression model with optimal hyperparameters
logreg = LogisticRegression(multi_class='multinomial', solver='sag', C=100, max_iter=500)

logreg.fit(xTrain_E_scaled, yTrain_E)

# NOTE: The original dataset for this script is ENGLAND
print("Original Dataset: England")

# Make predictions on the test set
yTest_pred = logreg.predict(xTest_E_scaled)

# Evaluate accuracy on the test set
selfAcc = accuracy_score(yTest_E, yTest_pred)
print("Self-Test Accuracy:", selfAcc)
# Calculate F1 score on the test set
selfF1 = f1_score(yTest_E, yTest_pred, average='weighted')
print("Test F1 Score:", selfF1)

# Evaluate accuracy on other league test sets
yTest_pred1 = logreg.predict(xTest_S_scaled)
acc1 = accuracy_score(yTest_S, yTest_pred1)
print("Test Accuracy Spain:", acc1)
f1_1 = f1_score(yTest_S, yTest_pred1, average='weighted')
print("Test F1 Score:", f1_1)

yTest_pred2 = logreg.predict(xTest_I_scaled)
acc2 = accuracy_score(yTest_I, yTest_pred2)
print("Test Accuracy Italy:", acc2)
f1_2 = f1_score(yTest_I, yTest_pred2, average='weighted')
print("Test F1 Score:", f1_2)

yTest_pred3 = logreg.predict(xTest_F_scaled)
acc3 = accuracy_score(yTest_F, yTest_pred3)
print("Test Accuracy France:", acc3)
f1_3 = f1_score(yTest_F, yTest_pred3, average='weighted')
print("Test F1 Score:", f1_3)

yTest_pred4 = logreg.predict(xTest_G_scaled)
acc4 = accuracy_score(yTest_G, yTest_pred4)
print("Test Accuracy Germany:", acc4)
f1_4 = f1_score(yTest_G, yTest_pred4, average='weighted')
print("Test F1 Score:", f1_4)

# Test Accuracy and F1 Score for other league test sets
data = {
    'Country': ['England', 'Spain', 'Italy', 'France', 'Germany'],
    'Accuracy': [selfAcc, acc1, acc2, acc3, acc4],
    'F1 Score': [selfF1, f1_1, f1_2, f1_3, f1_4]
}

df = pd.DataFrame(data)

# Plotting the matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.pivot_table(index='Country', values=['Accuracy', 'F1 Score']), annot=True, cmap='coolwarm')
plt.title('Accuracy vs. F1 Score Matrix (Original Country: England)')
plt.show()