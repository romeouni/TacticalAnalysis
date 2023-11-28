import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os



def main():

    appearances = pd.read_csv("data/appearances.csv")

    nonFeatures = ['gameID', 'playerID', 'position', 'positionOrder', 'leagueID', 'substituteIn', 'substituteOut']

    yTotal = appearances[["positionOrder", "leagueID"]]

    appearancesFeatures = appearances.drop(nonFeatures, axis=1)

    #recode positions to reduce number of categories

    # KEY:
    # 1: Goalkeepers
    # 2: Fullbacks
    # 3: Centerbacks
    # 4: Defensive Midfielders
    # 5: Midfielders
    # 6: Attacking Midfielders
    # 7: Wingers
    # 8: Strikers
    
    yTotal = yTotal.replace(4, 2)
    yTotal = yTotal.replace(5|6|7, 4)
    yTotal = yTotal.replace(8|9|10, 5)
    yTotal = yTotal.replace(11|12|13, 6)
    yTotal = yTotal.replace(14|15, 7)
    yTotal = yTotal.replace(16, 8)

    print(yTotal['positionOrder']==17)

    subs = yTotal.index[yTotal['positionOrder'] == 17].tolist()

    appearancesFeatures = appearancesFeatures.drop(index=subs)
    yTotal = yTotal.drop(index=subs)


    # Perform standard scaling on the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(appearancesFeatures)

    
    # Apply PCA to capture 95% of the variance
    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(X_scaled)

    # Get the loading values (coefficients of original features on principal components)
    loading_values = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Visualize the loading values for each principal component separately
    num_components = loading_values.shape[1]

    for i in range(num_components):
        sorted_indices = np.argsort(loading_values[:, i])
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(loading_values)), loading_values[sorted_indices, i], align='center')
        plt.yticks(range(len(loading_values)), appearancesFeatures.columns[sorted_indices])
        plt.xlabel('Loading Value')
        plt.title(f'Principal Component {i+1} Loadings')
        plt.tight_layout()
        plt.show()

    # Get explained variance ratios for each principal component
    explained_var_ratio = pca.explained_variance_ratio_

    # Create a bar chart of explained variance for each component
    num_components = len(explained_var_ratio)
    component_index = range(1, num_components + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(component_index, explained_var_ratio * 100, align='center', alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance of Principal Components')
    plt.xticks(component_index)
    plt.grid(axis='y')
    plt.show()

    appearances = np.delete(appearances, subs, axis=0)
    xPL = appearances.query('leagueID == 1').drop(nonFeatures, axis=1)
    xSA = appearances.query('leagueID == 2').drop(nonFeatures, axis=1)
    xBL = appearances.query('leagueID == 3').drop(nonFeatures, axis=1)
    xLL = appearances.query('leagueID == 4').drop(nonFeatures, axis=1)
    xLU = appearances.query('leagueID == 5').drop(nonFeatures, axis=1)
    xWhole = appearances.drop(nonFeatures, axis=1)

    yPL = yTotal.query('leagueID == 1').drop('leagueID', axis=1)
    ySA = yTotal.query('leagueID == 2').drop('leagueID', axis=1)
    yBL = yTotal.query('leagueID == 3').drop('leagueID', axis=1)
    yLL = yTotal.query('leagueID == 4').drop('leagueID', axis=1)
    yLU = yTotal.query('leagueID == 5').drop('leagueID', axis=1)
    yWhole = yTotal.drop('leagueID', axis=1)

    xPL = pd.DataFrame(pca.transform(xPL))
    xSA = pd.DataFrame(pca.transform(xSA))
    xBL = pd.DataFrame(pca.transform(xBL))
    xLL = pd.DataFrame(pca.transform(xLL))
    xLU = pd.DataFrame(pca.transform(xLU))
    xWhole = pd.DataFrame(pca.transform(xWhole))

    xPL.to_csv('data/xEngland.csv', index=False)
    xSA.to_csv('data/xItaly.csv', index=False)
    xBL.to_csv('data/xGermany.csv', index=False)
    xLL.to_csv('data/xSpain.csv', index=False)
    xLU.to_csv('data/xFrance.csv', index=False)
    xWhole.to_csv('data/xWhole.csv', index=False)

    yPL.to_csv('data/yEngland.csv', index=False)
    ySA.to_csv('data/yItaly.csv', index=False)
    yBL.to_csv('data/yGermany.csv', index=False)
    yLL.to_csv('data/ySpain.csv', index=False)
    yLU.to_csv('data/yFrance.csv', index=False)
    yWhole.to_csv('data/yWhole.csv', index=False)

if __name__ == "__main__":
    main()