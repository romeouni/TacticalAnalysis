import numpy as np
import pandas as pd



def main():
    appearances = pd.read_csv("data/appearances.csv")

    PL = appearances.where('leagueID == 1')
    SA = appearances.where('leagueID == 2')
    BL = appearances.where('leagueID == 3')
    LL = appearances.where('leagueID == 4')
    LU = appearances.where('leagueID == 5')

    PL.to_csv("/data")
    SA.to_csv("/data")
    BL.to_csv("/data")
    LL.to_csv("/data")
    LU.to_csv("/data")
    
    return None

if __name__ == "__main__":
    main()