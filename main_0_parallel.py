#%%
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import get_scorer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,ADASYN, RandomOverSampler,KMeansSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RandomUnderSampler, CondensedNearestNeighbour
from imblearn.combine import SMOTETomek, SMOTEENN
#%%
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

percent = 1
#%%
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

def model_eval(i,model,X1,y1,X2,y2,X3,y3, random_state):
    cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)  # , random_state=random_state)
    idr = (y1[(y1 == 1)].sample(int((y1 == 1).sum() * i), random_state=random_state)).sort_index()
    idr = pd.concat([idr, y1[y1 == 0]])
    Xt = X1.loc[idr.index, :]
    yt = y1.loc[idr.index]
    scores = sklearn.model_selection.cross_validate(model, Xt, yt, cv=cv, scoring=['f1_macro', 'balanced_accuracy'],
                                                n_jobs=5)
    trainin_samples_before = Xt.shape[0]
    model_t = clone(model)
    model_t.fit(Xt, yt)
    trainin_samples_after = model_t['pipelineinspector'].get_info()['n_samples']
    y2p = model_t.predict(X2)
    y3p = model_t.predict(X3)

    text = f" Percent={i}"
    res = {"percent":i,
           "training_samples_after":trainin_samples_after,
           "training_samples_before": trainin_samples_before,
           }
    for scorer_name in ['f1_macro', 'balanced_accuracy']:
        scorer = get_scorer(scorer_name)
        me = scores[f'test_{scorer_name}'].mean()
        std = scores[f'test_{scorer_name}'].std()
        sc2 = scorer._score_func(y2, y2p)
        sc3 = scorer._score_func(y3, y3p)
        text += f" ; {scorer_name} ; {me * 100:.2f} ; {std * 100:.2f} ; {sc2 * 100:.2f} ; {sc3 * 100:.2f}"
        res.update({scorer_name +"_mean":me,
        scorer_name +"_std":std,
        scorer_name +"_UT2":sc2,
        scorer_name +"_UT3":sc3})
    print(text)
    return text,res

# df1 = pd.read_csv(f"data/mb/data1_{percent * 100}.csv")
df1 = pd.read_csv(f"data/mb/dataA_1_100.csv")
X1 = df1.loc[:, cols_x_imp_vib]
y1 = df1.loc[:, 'stan']

# df2 = pd.read_csv(f"data/mb/dataA_2.csv")
df2 = pd.read_csv(f"data/mb/dataA_2.csv")
X2 = df2.loc[:, cols_x_imp_vib]
y2 = df2.loc[:, 'stan']

# df3 = pd.read_csv(f"data/mb/dataA_3.csv")
df3 = pd.read_csv(f"data/mb/dataA_3.csv")
X3 = df3.loc[:, cols_x_imp_vib]
y3 = df3.loc[:, 'stan']

# %%
all_all_res = []
for random_state in [31,33,11,97,142,57,297]:#[19, 21, 42, 1999, 2001]:
    print(f"========================== {random_state} =======================")
    mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(40, 20), max_iter=500, random_state=random_state)

    all_res = []
    models = []
    models += ["no Sampler()"]
    models += [RandomOverSampler(random_state=random_state),
               RandomUnderSampler(random_state=random_state)]
    #models += [CondensedNearestNeighbour()]
    models += [SMOTE(random_state=random_state, k_neighbors=nn) for nn in [3,5,7,9]]
    models += [BorderlineSMOTE(random_state=random_state, k_neighbors=nn) for nn in [3,5,7,9]]
    models += [EditedNearestNeighbours(n_neighbors=nn) for nn in [3,5,7,9]]
    models += [TomekLinks() ]
    models += [ADASYN(random_state=random_state, n_neighbors=nn) for nn in [3,5,7,9]]
    models += [SMOTEENN(random_state=random_state, smote=SMOTE(random_state=random_state, k_neighbors=nn),enn=EditedNearestNeighbours(n_neighbors=nn)) for nn in [3,5,7,9]]
    models += [SMOTETomek(random_state=random_state,  smote=SMOTE(random_state=random_state, k_neighbors=nn),tomek=TomekLinks()) for nn in [3,5,7,9]]


    for smp in models:
        if smp != "no Sampler()":
            model = make_pipeline(StandardScaler(),
                                  smp,
                                  PipelineInspector(),
                                  mlp)
        else:
            model = make_pipeline(StandardScaler(),
                                  PipelineInspector(),
                                  mlp)

        res = []
        res = Parallel(n_jobs=32)(delayed(model_eval)(i, clone(model), X1, y1, X2, y2, X3, y3, random_state) for i in [0.005, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1])

        resl = []
        for a,b in res:
    #        print(a)
            resl.append(b)
        df_res = pd.DataFrame(resl)
        df_res["Sampler"] = str(smp)
        df_res["Seed"] = random_state
        all_res.append(df_res)

    df_res = pd.concat(all_res)
    df_res.to_csv(f"results/res_16_seed_{random_state}.csv",index=False)
    all_all_res.append(df_res)

df_res_all = pd.concat(all_all_res)
df_res_all.to_csv(f"results/res_16_all.csv",index=False)
