import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#FULL TARGET PL
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

#FULL TARGET L1
    # Accuracy on Premier League: 0.3206083390293917
    # F1 Score on Premier League: 0.26294568501625754
    # Accuracy on Ligue 1: 0.3309934245601564
    # F1 Score on Ligue 1: 0.27305975292260715
    # Accuracy on Bundesliga: 0.31173736816416614
    # F1 Score on Bundesliga: 0.2583444892025933
    # Accuracy on Serie A: 0.3643261163041064
    # F1 Score on Serie A: 0.3182866995027068
    # Accuracy on La Liga: 0.3261846579743331
    # F1 Score on La Liga: 0.26819885446698444

#FULL TARGET BL
    # Accuracy on Premier League: 0.310850991114149
    # F1 Score on Premier League: 0.24633340897119613
    # Accuracy on Ligue 1: 0.3097264923847945
    # F1 Score on Ligue 1: 0.2460108418080554
    # Accuracy on Bundesliga: 0.30981432360742706
    # F1 Score on Bundesliga: 0.2478677710301022
    # Accuracy on Serie A: 0.3499547155624669
    # F1 Score on Serie A: 0.2976999294467179
    # Accuracy on La Liga: 0.31572651617423403
    # F1 Score on La Liga: 0.25257379370536687

#FULL TARGET SA
    # Accuracy on Premier League: 0.3226930963773069
    # F1 Score on Premier League: 0.24160809893506816
    # Accuracy on Ligue 1: 0.3239261405036521
    # F1 Score on Ligue 1: 0.24139875542792918
    # Accuracy on Bundesliga: 0.3146446533539885
    # F1 Score on Bundesliga: 0.24060761368398526
    # Accuracy on Serie A: 0.39166097060833904
    # F1 Score on Serie A: 0.31816894702001813
    # Accuracy on La Liga: 0.3251080845537347
    # F1 Score on La Liga: 0.24552362141202203

#FULL TARGET LL\
    # Accuracy on Premier League: 0.3172761449077239
    # F1 Score on Premier League: 0.25526380407229704
    # Accuracy on Ligue 1: 0.3250990776448844
    # F1 Score on Ligue 1: 0.26243898540114946
    # Accuracy on Bundesliga: 0.31477197971266685
    # F1 Score on Bundesliga: 0.25545435515840637
    # Accuracy on Serie A: 0.3621217040619286
    # F1 Score on Serie A: 0.3110501820893058
    # Accuracy on La Liga: 0.3233082706766917
    # F1 Score on La Liga: 0.2626911137079997
    
    

def main():
    leagueList = ["Premier Leage", "Ligue 1", "Bundesliga", "Serie A", "La Liga"]
    # Model trained on P1
    accs = [0.3179254955570745, 0.3238905969539178, 0.31362604248456166, .3671115364240674, 0.32456125360993865]
    F1 = [0.2525691945778334, 0.25561447680840826, 0.25196243803251694, 0.3128108345289699, 0.2579153125394861]

    # Number of leagues
    num_leagues = len(leagueList)

    # Create an array of indices for the number of leagues
    indices = np.arange(num_leagues)

    # Width of each bar
    bar_width = 0.35

    # Plotting the bar graph for accuracy scores
    plt.figure(figsize=(10, 6))
    plt.bar(indices, accs, bar_width, label='Accuracy', color='#81A684')

    # Plotting the bar graph for F1 scores with an offset
    plt.bar(indices + bar_width, F1, bar_width, label='F1 Score', color='#466060')

    # Set labels, title, and ticks
    plt.xlabel('Leagues')
    plt.ylabel('Scores')
    plt.title('Accuracy and F1 Scores from PL Model')
    plt.xticks(indices + bar_width / 2, leagueList)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('Visualizations/gb_england.png')
    plt.show()
    # Model trained on L1
    accs = [0.3206083390293917, 0.3309934245601564, 0.31173736816416614, 0.3643261163041064, 0.3261846579743331]
    F1 = [0.26294568501625754, 0.27305975292260715, 0.2583444892025933, 0.3182866995027068, 0.26819885446698444]

    # Plotting the bar graph for accuracy scores
    plt.figure(figsize=(10, 6))
    plt.bar(indices, accs, bar_width, label='Accuracy', color='#81A684')

    # Plotting the bar graph for F1 scores with an offset
    plt.bar(indices + bar_width, F1, bar_width, label='F1 Score', color='#466060')

    # Set labels, title, and ticks
    plt.xlabel('Leagues')
    plt.ylabel('Scores')
    plt.title('Accuracy and F1 Scores from L1 Model')
    plt.xticks(indices + bar_width / 2, leagueList)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('Visualizations/gb_france.png')
    plt.show()

    # Model trained on BL
    accs = [0.310850991114149, 0.3097264923847945, 0.30981432360742706, 0.3499547155624669, 0.31572651617423403]
    F1 = [0.24633340897119613,0.2460108418080554, 0.2478677710301022, 0.2976999294467179, 0.25257379370536687]

    # Plotting the bar graph for accuracy scores
    plt.figure(figsize=(10, 6))
    plt.bar(indices, accs, bar_width, label='Accuracy', color='#81A684')

    # Plotting the bar graph for F1 scores with an offset
    plt.bar(indices + bar_width, F1, bar_width, label='F1 Score', color='#466060')

    # Set labels, title, and ticks
    plt.xlabel('Leagues')
    plt.ylabel('Scores')
    plt.title('Accuracy and F1 Scores from BL Model')
    plt.xticks(indices + bar_width / 2, leagueList)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('Visualizations/gb_germany.png')
    plt.show()

    # Model trained on SA
    accs = [0.3226930963773069, 0.3239261405036521, 0.3146446533539885, 0.39166097060833904, 0.3251080845537347]
    F1 = [0.24160809893506816, 0.24139875542792918, 0.24060761368398526, 0.31816894702001813, 0.24552362141202203]

    # Plotting the bar graph for accuracy scores
    plt.figure(figsize=(10, 6))
    plt.bar(indices, accs, bar_width, label='Accuracy', color='#81A684')

    # Plotting the bar graph for F1 scores with an offset
    plt.bar(indices + bar_width, F1, bar_width, label='F1 Score', color='#466060')

    # Set labels, title, and ticks
    plt.xlabel('Leagues')
    plt.ylabel('Scores')
    plt.title('Accuracy and F1 Scores from SA Model')
    plt.xticks(indices + bar_width / 2, leagueList)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('Visualizations/gb_italy.png')
    plt.show()

    # Model trained on LL
    accs = [0.3172761449077239, 0.3250990776448844, 0.31477197971266685, 0.3621217040619286, 0.3233082706766917]
    F1 = [0.25526380407229704, 0.26243898540114946, 0.25545435515840637, 0.3110501820893058, 0.2626911137079997]

    # Plotting the bar graph for accuracy scores
    plt.figure(figsize=(10, 6))
    plt.bar(indices, accs, bar_width, label='Accuracy', color='#81A684')

    # Plotting the bar graph for F1 scores with an offset
    plt.bar(indices + bar_width, F1, bar_width, label='F1 Score', color='#466060')

    # Set labels, title, and ticks
    plt.xlabel('Leagues')
    plt.ylabel('Scores')
    plt.title('Accuracy and F1 Scores from PL Model')
    plt.xticks(indices + bar_width / 2, leagueList)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('Visualizations/gb_spain.png')
    plt.show()

if __name__=="__main__":
    main()