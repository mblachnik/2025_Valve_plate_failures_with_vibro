#%%

import numpy as np
import pandas as pd
import os

fNames = ['dane_OT.csv',   #Records representing normal working conditions
          'dane_UT1.csv',  #Records representing Failure 1
          'dane_UT2.csv',  #Records representing Failure 2
          'dane_UT3.csv']  #Records representing Failure 3
#Loadeing all datasets and preparing classification problems
dfs = []
dirname = 'data/2024'
for f in fNames:
    df = pd.read_csv(os.path.join(dirname, f))
    df.Czas = pd.to_datetime(df.Czas)
    df.set_index(df.Czas, inplace=True)
    dfs.append(df)
#Columns in the dataset
cols =  ['Czas2', #0 Time2
         'Czas', #1 Time
         'Pressure - leak line', #2
         'Temperature - leak line', #3
         'Pressure - output', #4
         'Temperature - suction line', #5
        'Temperature - output', #6
         'Flow - leak line', #7
         'Flow - output',#8
        'Temp. diff', #9
         'stan'] #Output column indicating wheter given sample represent normal roking conditions or failure state

#Input columns
cols_x = ['Pressure - leak line', #2
         'Temperature - leak line', #3
         'Pressure - output', #4
         'Temperature - suction line', #5
        'Temperature - output', #6
         'Flow - leak line', #7
         'Flow - output',#8
        'Temp. diff', #9
         ]
#Labels
col_y =  "stan"

#From dane_OT.csv remove samples with incorrect temperature
dfs[0] = dfs[0].iloc[np.r_[0:2447,2449:dfs[0].shape[0]]]

#Dataframe containing Failure 2
data_UT2 = pd.concat([
            dfs[0].iloc[-15000:,:], #Last 15000 samples from dane_OT.csv are used as negative class (Class 0)
            dfs[2],
        ],axis=0)

#Dataframe containing Failure 3
data_UT3 = pd.concat([
            dfs[0].iloc[-15000:,:], #Last 15000 samples from dane_OT.csv are used as negative class (Class 0) (same as in Failure 2)
            dfs[3],
        ], axis=0)

#Dataframe containing Failure 1
data_UT1 = pd.concat([
            dfs[0].iloc[:-15000,:], #All samples except last 15000 samples from dane_OT.csv are used as negative class (Class 0)
            dfs[1],
        ], axis=0)
data_UT1 = data_UT1.loc[~np.any(data_UT1.isna(), axis=1), :] #Removing nan's

X1 = data_UT1.loc[:,cols_x].to_numpy()
y1 = data_UT1.loc[:,col_y].to_numpy()

X2 = data_UT2.loc[:,cols_x].to_numpy()
y2 = data_UT2.loc[:,col_y].to_numpy()

X3 = data_UT3.loc[:,cols_x].to_numpy()
y3 = data_UT3.loc[:,col_y].to_numpy()

data_UT1.to_csv("data/mb/data1_2024.csv", index=False)
data_UT2.to_csv("data/mb/data2_2024.csv", index=False)
data_UT3.to_csv("data/mb/data3_2024.csv", index=False)