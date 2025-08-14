#%%
import pandas as pd
import numpy as np
import sklearn
import matplotlib

from sklearn.metrics import get_scorer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

from scipy.integrate import simpson
from scipy.stats import rankdata

# matplotlib.use('qtagg')
%matplotlib qt
import matplotlib.pyplot as plt
#%% Read results file
#List of models to display
models = {"SMOTE",
          "BorderlineSMOTE",
          "EditedNearestNeighbours",
          "TomekLinks",
          "ADASYN",
          "SMOTEENN",
          "SMOTETomek",
          "no Sampler",
          "RandomOverSampler",
          "RandomUnderSampler"}

#df_load = pd.read_csv("results/res_15_all.csv")
df_load = pd.read_csv("results/res_15_16_all.csv")

#Missing: average over random_seed
# df = df_load.groupby(['Sampler', 'percent']).mean()
# df.reset_index(inplace=True)
df = df_load.loc[df_load["Seed"]==19,:]

#Generate new column Sampler_type
df.loc[:,"Sampler_type"] = df["Sampler"].str.split("(", n=1).str[0]

#Get indeces with the highest f1_macro on UT1
max_indices = df.groupby(['Sampler_type', 'percent'])['f1_macro_mean'].idxmax()
max_indices.dropna(inplace=True)

# Use these indices to get the full rows including results on UT2 and UT3
res = df.loc[max_indices].reset_index(drop=True)


#%% Plot results

area = []

#For each dataset
for i,col_y in enumerate(["f1_macro_mean","f1_macro_UT2", "f1_macro_UT3"]):
    plt.figure(i, clear=True)
    plt.title(col_y)
    col_x = 'percent'
    #For each sampler model
    for model in models:
        #Get subset of sample for a given sampler
        m = res[res["Sampler_type"] == model]
        #Plot the results and calculate area under curve
        # m.drop(m[m['percent']==1].index, inplace=True)
        x = m.loc[:,col_x]
        y = m.loc[:,col_y]
        plt.plot(x, y, label = f'{model}')
        area.append({
            "dataset f1":col_y,
            "model":m['Sampler_type'][m.index.min()],
            # "area trapeze":np.trapz(y, x),
            "area simpson":simpson(y, x),
            })
    plt.legend()
    plt.show()

area_df = pd.DataFrame(area)

rank_results = []
for col_y in area_df['dataset f1'].unique().tolist():
    area_df.loc[area_df['dataset f1']==col_y,col_y+'_rank'] = rankdata(-area_df.loc[area_df['dataset f1']==col_y]['area simpson'], method='average',)

for model in models:
    r1 = area_df.loc[area_df["model"] == model, 'f1_macro_mean_rank'].max()
    r2 = area_df.loc[area_df["model"] == model, 'f1_macro_UT2_rank'].max()
    r3 = area_df.loc[area_df["model"] == model, 'f1_macro_UT3_rank'].max()
    rank_results.append({
            'model': model,
            'UT1 rank': r1,
            'UT2 rank': r2,
            'UT3 rank': r3,
            'average rank': np.mean([r1,r2,r3])
        })
r_res_df = pd.DataFrame(rank_results)

area_df.to_csv('results/areas_under_curves.csv')
r_res_df.to_csv('results/ranks.csv')
