import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data = pd.read_csv("data/appearances.csv")


    # Selecting the specific columns
    selected_columns = ['xGoals', 'xAssists', 'xGoalsBuildup']
    selected_data = data[selected_columns]

    # Creating the horizontal violin plot
    plt.figure(figsize=(10, 6))  # Set the figure size as per your preference
    sns.violinplot(data=selected_data, orient='h', palette='viridis')
    plt.xlabel('Value')
    plt.title('Violin Plot of Goals, Assists, and xGoalsBuildup')

    # Setting x-axis limits
    plt.xlim(-0.1, 1.5)
    plt.savefig("Visualizations/selectedDisplay.png")
    plt.show()
    plt.close()

    median_xGoalsBuildup = data.groupby('position')['xGoalsBuildup'].median().sort_values()

    # Creating a bar plot
    plt.figure(figsize=(10, 6))  # Set the figure size as per your preference
    median_xGoalsBuildup.plot(kind='barh', color='skyblue')
    plt.xlabel('Median xGoalsBuildup')
    plt.ylabel('Position')
    plt.title('Median xGoalsBuildup by Position')
    plt.savefig("Visualizations/medianXGB.png")
    plt.show()
    plt.close()

    # Selecting the columns of interest
    selected_columns = ["goals", "ownGoals", "shots", "xGoals", "xGoalsChain", "xGoalsBuildup",
                        "assists", "keyPasses", "xAssists", "yellowCard", "redCard"]

    # Creating a correlation matrix
    correlation_matrix = data[selected_columns].corr()

    # Creating a heatmap for the correlation matrix
    plt.figure(figsize=(10, 8))  # Set the figure size as per your preference
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Offensive Stats')
    plt.savefig("Visualizations/heatmap.png")
    plt.show()

if __name__=="__main__":
    main()