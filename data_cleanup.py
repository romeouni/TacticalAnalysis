import numpy as np
import pandas as pd
import os



def main():

    appearances = pd.read_csv("data/appearances.csv")

    PL = appearances.query('leagueID == 1')
    SA = appearances.query('leagueID == 2')
    BL = appearances.query('leagueID == 3')
    LL = appearances.query('leagueID == 4')
    LU = appearances.query('leagueID == 5')

    PL.to_csv('data/england.csv', index=False)
    SA.to_csv('data/italy.csv', index=False)
    BL.to_csv('data/germany.csv', index=False)
    LL.to_csv('data/spain.csv', index=False)
    LU.to_csv('data/france.csv', index=False)

if __name__ == "__main__":
    main()