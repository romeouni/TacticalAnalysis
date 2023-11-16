import pandas as pd
import numpy as np


def main():
    appearances = pd.read_csv("data/appearances.csv")

    PL = appearances.where('leagueID == 1')
    SA = appearances.where('leagueID == 2')
    BL = appearances.where('leagueID == 3')
    LL = appearances.where('leagueID == 4')
    LU = appearances.where('leagueID == 5')
    
    return None

if __name__ == "__main__":
    main()