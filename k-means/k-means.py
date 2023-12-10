import argparse
import numpy as np
import pandas as pd
import math
import statistics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns


def main():
    """
    Main file to run from the command line.
    """
    # Set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--x",
                        default="xWhole_k-means.csv",
                        help="filename for features of the training data")
    

    args = parser.parse_args()

    all_data = pd.read_csv(args.x)


    all_data_one_league = all_data.query('leagueID == 1') # gives me one league's Data to train the model on
    all_data_one_league = all_data_one_league.drop(columns=['leagueID'])
    all_data_one_league = all_data_one_league.drop(columns=['positionOrder'])

    k = 16  # Choose the number of clusters (we have 16 positions w/out subs)
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the model to the league I am training on
    kmeans.fit(all_data_one_league)    

# These lines allow me to separate the data based on each league. I uncomment one of them each time to test on a different league.

    all_data = all_data.query('leagueID == 1') # gives me England Data
    # all_data = all_data.query('leagueID == 2') # gives me Italy Data
    # all_data = all_data.query('leagueID == 3') # gives me Germany Data
    # all_data = all_data.query('leagueID == 4') # gives me Spain Data
    # all_data = all_data.query('leagueID == 5') # gives me France Data

    # Drop the position so it's not used to predict the clusters (this would be cheating!)
    all_data_without_position = all_data.drop(columns=['positionOrder'])
    all_data_without_position = all_data_without_position.drop(columns=['leagueID'])
    

    # Predict a model based on the fitted model and get the cluster labels for the prediction
    cluster_labels = kmeans.predict(all_data_without_position)

    # Calculate silhouette scores for the different models
    silhouette_val = silhouette_score(all_data_without_position, cluster_labels)
    print("Checking now")
    print(" ")
    print(" ")
    print("Silhouette score: ", silhouette_val)
    print(" ")
    print(" ")

    # Add cluster labels as a new column to the DataFrame
    all_data['Cluster'] = cluster_labels

   
    # Keep track of the most common cluster for each position
    most_common_clustering_by_position = all_data.groupby('positionOrder')['Cluster'].agg(lambda x: x.mode().iloc[0]).reset_index()


    # Plot a bar plot to visualize the most common cluster for each position

    custom_colors = ['#81A684', '#466060']

    plt.figure(figsize=(10, 6))
    sns.barplot(x='positionOrder', y='Cluster', data=most_common_clustering_by_position, palette=custom_colors)
    plt.title(f'Most Common Cluster for Each Position')
    plt.xlabel('Position')
    plt.ylabel('Most Common Cluster')
    plt.show()





if __name__ == "__main__":
    main()
