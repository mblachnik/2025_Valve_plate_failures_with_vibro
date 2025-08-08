
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import pickle
# import datetime
# import time
# import os
# import copy


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# %matplotlib qt
percent=1
cols =  [
         'Applied torque',
         'Pressure - leak line', #2
         'Temperature - leak line', #3
         'Pressure - output', #4
         'Temperature - suction line', #5
         'Temperature - output', #6
         'Flow - leak line', #7
         'Flow - output',#8
          'Sensor 1',#9
          'Sensor 2',#10
          'Sensor 3',#11
         'Temp. diff', #12
         'stan',
         # 'hPower',
         # 'Fleak_mul_Pout'
         ]

cols_x = [
         # 'Applied torque',
         # 'Pressure - leak line', #2 0
         'Temperature - leak line', #3 1
         'Pressure - output', #4 2
         'Temperature - suction line', #5 3
         'Temperature - output', #6 4
          'Flow - leak line', #7 5
          'Flow - output',#8 6
          #   'Sensor 1',#9 7
          #   'Sensor 2',#10 8
          #   'Sensor 3',#11 9
          'Temp. diff', #12 10
            # 'hPower',
            # 'Fleak_mul_Pout'
         ]

cols_x_vib = [
         # 'Applied torque',
         # 'Pressure - leak line', #2 0
         'Temperature - leak line', #3 1
         'Pressure - output', #4 2
         'Temperature - suction line', #5 3
         'Temperature - output', #6 4
          'Flow - leak line', #7 5
          'Flow - output',#8 6
            'Sensor 1',#9 7
            'Sensor 2',#10 8
            'Sensor 3',#11 9
          'Temp. diff', #12 10
            # 'hPower',
            # 'Fleak_mul_Pout'
         ]

cols_x_imp = [
         # 'Applied torque',
          'Pressure - leak line', #2 0
           'Temperature - leak line', #3 1
           'Pressure - output', #4 2
           'Temperature - suction line', #5 3
           'Temperature - output', #6 4
           'Flow - leak line', #7 5
            'Flow - output',#8 6
              # 'Sensor 1',#9 7
         #  #   'Sensor 2',#10 8
         #  #   'Sensor 3',#11 9
            'Temp. diff', #12 10
            # 'hPower',
            # 'Fleak_mul_Pout'
         ]

cols_x_imp_vib = [
         # 'Applied torque',
          'Pressure - leak line', #2 0
          'Temperature - leak line', #3 1
           'Pressure - output', #4 2
          'Temperature - suction line', #5 3
           'Temperature - output', #6 4
            'Flow - leak line', #7 5
           'Flow - output',#8 6
             'Sensor 1',#9 7
             'Sensor 2',#10 8
             'Sensor 3',#11 9
            'Temp. diff', #12 10
            # 'hPower',
            # 'Fleak_mul_Pout'
         ]

cols_x_vib_only = [
         # 'Applied torque',
         # 'Pressure - leak line', #2 0
         # 'Temperature - leak line', #3 1
          # 'Pressure - output', #4 2
         # 'Temperature - suction line', #5 3
          # 'Temperature - output', #6 4
           # 'Flow - leak line', #7 5
         #  'Flow - output',#8 6
             'Sensor 1',#9 7
             'Sensor 2',#10 8
             'Sensor 3',#11 9
           # 'Temp. diff', #12 10
            # 'hPower',
            # 'Fleak_mul_Pout'
         ]

col_y =  "stan"

cols_list = []
# cols_list.append(cols_x)
# cols_list.append(cols_x_vib)
cols_list.append(cols_x_imp)
cols_list.append(cols_x_imp_vib)
cols_list.append(cols_x_vib_only)

# %% Load data files
fNames = ['dane_OT.csv',
          'dane_UT1.csv',
          'dane_UT2.csv',
          'dane_UT3.csv']
dfs = []
temps = pd.DataFrame()
fleak = pd.DataFrame()
torque = pd.DataFrame()

for f in fNames:

    df = pd.read_csv('data/data250505/' + f)
    temps = pd.concat([temps,pd.DataFrame(([(df["Temperature - suction line"].min(), df["Temperature - suction line"].max())]))])
    # fleak = pd.concat([fleak,pd.DataFrame(([(df["Flow - leak line"].min(), df["Flow - leak line"].max())]))])  # print(f, ' temp min - ', df['Temperature - suction line'].min())
    # torque = pd.concat([torque,pd.DataFrame(([(df["Applied torque"].min(), df["Applied torque"].max())]))])  # print(f, ' temp min - ', df['Temperature - suction line'].min())

    df = df.reset_index(drop=True)
    dfs.append(df)

    
#%% Make dfs[x]

           
for i in range(len(dfs)):
    dfs[i] = dfs[i][
            (dfs[i]['Applied torque']>19)
            & (dfs[i]['Applied torque']<221)
            & (dfs[i]['Temperature - suction line']> temps[0].max()) 
            & (dfs[i]['Temperature - suction line']< temps[1].min()) 
            & (dfs[i]['Temp. diff']>0)
            & (dfs[i]['Flow - output']>55)
            & (dfs[i]['Sensor 1']<0.002)
            & (dfs[i]['Sensor 2']<0.002)
            & (dfs[i]['Sensor 3']<0.01)
            # & (dfs[i]['Temperature - suction line']< temps[1].min()) 
            # & (dfs[i]['Sensor1']< temps[1].min()) 
            # & (dfs[i]['Temperature - suction line']< temps[1].min()) 
            ]
    dfs[i].dropna(inplace = True)
    dfs[i].drop_duplicates(inplace=True)
    dfs[i] = dfs[i].reset_index(drop=True)
# data_UT1 = pd.DataFrame()
# data_UT1 = dfs[0]
# data_UT2 = dfs[1]
# data_UT3 = dfs[2]




#%% Podział danych OT na częsci i zbiory treningowe i testowe

df0train = dfs[0][dfs[0]['Time'].str.contains('2024-01-05')]
df0test = dfs[0][dfs[0]['Time'].str.contains('2024-01-04')]

#%% Wykres temperatury df0train i df0est

plt.figure('Train/test')
# plt.title('Train/test - temp na ssaniu')
plt.title('Dataset OT - oil temperature in suction line')
# plt.plot(df0train['Temperature - suction line'],'.')
# plt.plot(df0train['Flow - output'],'.')
# plt.plot(df0train['Flow - leak line'],'.')
plt.plot(df0test['Temperature - suction line'],'.', color='blue', label='OT_test')
plt.plot(df0train['Temperature - suction line'],'.', color='orange', label='OT_train')
plt.legend()
plt.show()

#%% Tworzenie zbiorów UT1...UT3
# dft=dfs[1].sample(round(df0train.shape[0]*0.48))
# dft=dfs[1].sample(round(df0train.shape[0]*0.25))
dft=dfs[1].sample(round(df0train.shape[0]*percent))
dft = dft.sort_index()
data_UT1 = pd.concat([
            df0train,
            dft
        ], axis=0)
          
# temp diff <-8
# flow_output < -4
# sensor1 > 5
# sensor2 > 5
# sensor3 >5
dft = dfs[2].sample(df0test.shape[0])
dft = dft.sort_index()
data_UT2 = pd.concat([
            df0test,
            dft
        ],axis=0)

dft = dfs[3].sample(df0test.shape[0])
dft = dft.sort_index()
data_UT3 = pd.concat([
            df0test,
            dft
        ], axis=0)

data_UT1.reset_index(inplace=True)
data_UT2.reset_index(inplace=True)
data_UT3.reset_index(inplace=True)

print('\nData UT1: \n',data_UT1['stan'].value_counts(sort=False))

print('\nData UT2: \n', data_UT2['stan'].value_counts(sort=False))

print('\nData UT3: \n', data_UT3['stan'].value_counts(sort=False))

# plt.plot(data_UT1['Applied torque'].reset_index(),'.')
# plt.figure()
# plt.plot(data_UT1.index, data_UT1['Applied torque'],'.')
# plt.plot(data_UT1.index, data_UT1['stan']*200,'.')

#%% Wykresy cech
#%matplotlib qt
# folder=''
plt.figure('Cechy',figsize=(12,8))

for n in range(1, len(cols)-1):
    plt.subplot(3,4,n)
    plt.title(cols[n])
    plt.plot(data_UT1[data_UT1['stan']==0][cols[n]],'.', label='OT')
    # plt.plot(dfs[1][dfs[1]['stan']==1][cols[n]],'.', label='OT')
    plt.plot(data_UT1[data_UT1['stan']==1][cols[n]],'.', label='UT1')

    # plt.plot(X_UT1_std_df[y_UT1==0][n],'x', label='OT')
    # plt.plot(X_UT1_std_df[y_UT1==1][n],'.', label='UT1')
    plt.legend()
    plt.grid(True)

    # fig_name=str(np.random.randn(1))
    # plt.savefig('cechy_v4/'+cols[n]+fig_name+'.jpg')
    # plt.savefig('cechy_v4/pdf/'+cols[n]+fig_name+'.pdf')
    # plt.savefig(folder+'/'+fig_name+'.pdf')
    # plt.close()
plt.tight_layout()
plt.show()    
#%%
plt.figure('Cechy UT2',figsize=(12,8))

for n in range(1, len(cols)-1):
    plt.subplot(3,4,n)
    plt.title(cols[n])
    plt.plot(data_UT2[data_UT2['stan']==0][cols[n]],'.', label='OT')
    # plt.plot(dfs[1][dfs[1]['stan']==1][cols[n]],'.', label='OT')
    plt.plot(data_UT2[data_UT2['stan']==1][cols[n]],'.', label='UT1')

    # plt.plot(X_UT1_std_df[y_UT1==0][n],'x', label='OT')
    # plt.plot(X_UT1_std_df[y_UT1==1][n],'.', label='UT1')
    plt.legend()
    plt.grid(True)

    # fig_name=str(np.random.randn(1))
    # plt.savefig('cechy_v4/'+cols[n]+fig_name+'.jpg')
    # plt.savefig('cechy_v4/pdf/'+cols[n]+fig_name+'.pdf')
    # plt.savefig(folder+'/'+fig_name+'.pdf')
    # plt.close()
plt.tight_layout()
plt.show()    
#%%
plt.figure('Cechy UT3',figsize=(12,8))

for n in range(1, len(cols)-1):
    plt.subplot(3,4,n)
    plt.title(cols[n])
    plt.plot(data_UT3[data_UT3['stan']==0][cols[n]],'.', label='OT')
    # plt.plot(dfs[1][dfs[1]['stan']==1][cols[n]],'.', label='OT')
    plt.plot(data_UT3[data_UT3['stan']==1][cols[n]],'.', label='UT1')

    # plt.plot(X_UT1_std_df[y_UT1==0][n],'x', label='OT')
    # plt.plot(X_UT1_std_df[y_UT1==1][n],'.', label='UT1')
    plt.legend()
    plt.grid(True)

    # fig_name=str(np.random.randn(1))
    # plt.savefig('cechy_v4/'+cols[n]+fig_name+'.jpg')
    # plt.savefig('cechy_v4/pdf/'+cols[n]+fig_name+'.pdf')
    # plt.savefig(folder+'/'+fig_name+'.pdf')
    # plt.close()
plt.tight_layout()
plt.show()    
#%% Zapis danych do plików v4
plt.close('all')

# data_UT1.to_csv('data_UT1_v5.csv')
# data_UT2.to_csv('data_UT2_v5.csv')
# data_UT3.to_csv('data_UT3_v5.csv')

# data_UT1.to_csv('data_UT1_v5_100.csv')
# data_UT1.to_csv('data_UT1_v5_005.csv')

# data_UT1.to_csv('data_UT1_v5_025.csv')
# data_UT1.to_csv('data_UT1_v5_010.csv')
# data_UT1.to_csv('data_UT1_v5_001.csv')
# modelsdf.to_csv('models.csv', sep=';')
#%%
data_UT1.to_csv(f'data/mb/data1_{percent*100}.csv',index=False)
data_UT2.to_csv('data/mb/data2.csv',index=False)
data_UT3.to_csv('data/mb/data3.csv',index=False)