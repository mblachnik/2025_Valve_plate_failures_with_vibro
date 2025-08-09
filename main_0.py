import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import get_scorer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier

percent = 1

cols_x_imp_vib = [
         # 'Applied torque',
          'Pressure - leak line', #2 0
          'Temperature - leak line', #3 1
           'Pressure - output', #4 2
          'Temperature - suction line', #5 3
           'Temperature - output', #6 4
            'Flow - leak line', #7 5
           'Flow - output',#8 6
            # 'Sensor 1',#9 7
            # 'Sensor 2',#10 8
            # 'Sensor 3',#11 9
            'Temp. diff', #12 10
            # 'hPower',
            # 'Fleak_mul_Pout'
         ]

df1 = pd.read_csv(f"data/mb/data1_{percent * 100}.csv")
#df1 = pd.read_csv(f"data/mb/data1_2024.csv")
X1 = df1.loc[:, cols_x_imp_vib]
y1 = df1.loc[:, 'stan']

df2 = pd.read_csv(f"data/mb/dataA_2.csv")
#df2 = pd.read_csv(f"data/mb/data2_2024.csv")
X2 = df2.loc[:, cols_x_imp_vib]
y2 = df2.loc[:, 'stan']

df3 = pd.read_csv(f"data/mb/dataA_3.csv")
#df3 = pd.read_csv(f"data/mb/data3_2024.csv")
X3 = df3.loc[:, cols_x_imp_vib]
y3 = df3.loc[:, 'stan']



#%%
random_state=21

model = make_pipeline(
    StandardScaler(),
           KNeighborsClassifier(n_neighbors=20)
           #sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(40,20), max_iter=500, random_state=random_state)
)

cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False)#, random_state=random_state)
res = []
for i in [1]:
    idr = (y1[(y1 == 1)].sample(int((y1 == 1).sum() * i), random_state=random_state)).sort_index()
    idr = pd.concat([idr, y1[y1 == 0]])
    Xt = X1.loc[idr.index, :]
    yt = y1.loc[idr.index]
    scores = sklearn.model_selection.cross_validate(model,Xt,yt, cv=cv,scoring=['f1_macro','balanced_accuracy'],n_jobs=4)

    model_t = clone(model)
    model_t.fit(Xt,yt)
    y2p = model_t.predict(X2)
    y3p = model_t.predict(X3)

    res.append(scores)
    text = f" Percent={i}"
    for scorer_name in ['f1_macro', 'balanced_accuracy']:
        scorer = get_scorer(scorer_name)
        me = scores[f'test_{scorer_name}'].mean()
        std = scores[f'test_{scorer_name}'].std()
        sc2 = scorer._score_func(y2,y2p)
        sc3 = scorer._score_func(y3, y3p)
        text += f" ; {scorer_name} ; {me * 100:.2f} ; {std * 100:.2f} ; {sc2 * 100:.2f} ; {sc3 * 100:.2f}"
    print(text)