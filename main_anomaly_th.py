import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import get_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.ensemble import IsolationForest

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def correct_labels(yp):
    yp = np.copy(yp)
    yp[yp == 1] = 0
    yp[yp == -1] = 1
    return yp

def correct_labels_df(yp,th=0):
    yp = np.copy(yp)
    yp[yp >= th] = 0
    yp[yp < th] = 1
    return yp

percent = 1

cols_x_imp_vib = [
    # 'Applied torque',
    'Pressure - leak line',  # 2 0
    'Temperature - leak line',  # 3 1
    'Pressure - output',  # 4 2
    'Temperature - suction line',  # 5 3
    'Temperature - output',  # 6 4
    'Flow - leak line',  # 7 5
    'Flow - output',  # 8 6
    'Sensor 1',#9 7
    'Sensor 2',#10 8
    'Sensor 3',#11 9
    'Temp. diff',  # 12 10
    # 'hPower',
    # 'Fleak_mul_Pout'
]

df1 = pd.read_csv(f"data/mb/data1_{percent * 100}.csv")
# df1 = pd.read_csv(f"data/mb/data1_2024.csv")
X1 = df1.loc[:, cols_x_imp_vib].to_numpy()
y1 = df1.loc[:, 'stan'].to_numpy()

df2 = pd.read_csv(f"data/mb/dataA_2.csv")
# df2 = pd.read_csv(f"data/mb/data2_2024.csv")
X2 = df2.loc[:, cols_x_imp_vib].to_numpy()
y2 = df2.loc[:, 'stan'].to_numpy()

df3 = pd.read_csv(f"data/mb/dataA_3.csv")
# df3 = pd.read_csv(f"data/mb/data3_2024.csv")
X3 = df3.loc[:, cols_x_imp_vib].to_numpy()
y3 = df3.loc[:, 'stan'].to_numpy()

# %%
random_state = 21

model = IsolationForest(n_estimators=300, random_state=random_state, n_jobs=4)
cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False)  # , random_state=random_state)
res = []

Xtr = X1[y1==0, :]

model.fit(Xtr)
y1p = model.decision_function(X1)
mx = y1p.max()
mi = y1p.min()
f1s = []
ths = []
for th in np.linspace(mi,mx,100):
    y1b = correct_labels_df(y1p, th=th)
    f1 = f1_score(y_true=y1,y_pred=y1b,average="macro")
    f1s.append(f1)
    ths.append(th)
    print(f1)

i = np.argmax(f1s)
th = ths[i]
print(th)
#%%
y2p = model.decision_function(X2)
y3p = model.decision_function(X3)
y1b = correct_labels_df(y1p,th=th)
y2b = correct_labels_df(y2p,th=th)
y3b = correct_labels_df(y3p,th=th)
text = ""
for scorer_name in ['f1_macro', 'balanced_accuracy']:
    scorer = get_scorer(scorer_name)
    sc1 = scorer._score_func(y1, y1b)
    sc2 = scorer._score_func(y2, y2b)
    sc3 = scorer._score_func(y3, y3b)
    text += f" ; {scorer_name} ;  {sc1 * 100:.2f} ; {sc2 * 100:.2f} ; {sc3 * 100:.2f}"
print(text)