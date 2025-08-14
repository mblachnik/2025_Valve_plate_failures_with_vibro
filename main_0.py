#%%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import get_scorer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


class PipelineInspector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to inspect data at any point in the pipeline
    """

    def __init__(self, step_name="", verbose=False):
        self.step_name = step_name
        self.verbose = verbose
        self.n_samples_ = None
        self.n_features_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Store information about the data at this step
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1] if hasattr(X, 'shape') else len(X[0])

        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(self.n_features_)]

        if self.verbose:
            print(f"[{self.step_name}] Samples: {self.n_samples_}, Features: {self.n_features_}")

        return X

    def get_info(self):
        """Get stored information about the data"""
        return {
            'n_samples': self.n_samples_,
            'n_features': self.n_features_,
            'feature_names': self.feature_names_
        }


from sklearn.base import clone
from sklearn.neural_network import MLPClassifier

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
            'Sensor 1',#9 7
            'Sensor 2',#10 8
            'Sensor 3',#11 9
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
            PipelineInspector(),
            SMOTE(),
            PipelineInspector(),
            MLPClassifier(hidden_layer_sizes=(40,20), max_iter=500, random_state=random_state)
)
#%%
cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False)#, random_state=random_state)
res = []
for i in [0.005]:#,0.01,0.03,0.05,0.1,0.25,0.5,1]:
    idr = (y1[(y1 == 1)].sample(int((y1 == 1).sum() * i), random_state=random_state)).sort_index()
    idr = pd.concat([idr, y1[y1 == 0]])
    Xt = X1.loc[idr.index, :]
    yt = y1.loc[idr.index]
    scores = sklearn.model_selection.cross_validate(model,Xt,yt, cv=cv,scoring=['f1_macro','balanced_accuracy'],n_jobs=4)

    model_t = clone(model)
    model_t.fit(Xt,yt)

    trainin_samples_before = model_t['pipelineinspector-1'].get_info()['n_samples']
    trainin_samples_after = model_t['pipelineinspector-2'].get_info()['n_samples']

    y2p = model_t.predict(X2)
    y3p = model_t.predict(X3)

    scores.update({"percent":i,
           "training_samples_after":trainin_samples_after,
           "training_samples_before": trainin_samples_before,
           })

    text = f" Percent={i}"
    for scorer_name in ['f1_macro', 'balanced_accuracy']:
        scorer = get_scorer(scorer_name)
        me = scores[f'test_{scorer_name}'].mean()
        std = scores[f'test_{scorer_name}'].std()
        sc2 = scorer._score_func(y2,y2p)
        sc3 = scorer._score_func(y3, y3p)
        text += f" ; {scorer_name} ; {me * 100:.2f} ; {std * 100:.2f} ; {sc2 * 100:.2f} ; {sc3 * 100:.2f}"
        scores.update({f'{scorer_name}_mean': me,
                       f'{scorer_name}_std': std,
                       f'{scorer_name}_UT1': sc2,
                       f'{scorer_name}_UT2': sc3,
                       })

    res.append(scores)
    print(text)

df_res = pd.DataFrame(res)
df_res.to_csv(f"results/res_test.csv")