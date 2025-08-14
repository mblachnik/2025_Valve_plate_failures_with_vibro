#%%
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
import matplotlib



matplotlib.use('TkAgg')
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

#For each dataset
for i,col_y in enumerate(["f1_macro_mean","f1_macro_UT2", "f1_macro_UT3"]):
    plt.figure(i, clear=True)
    plt.title(col_y)
    col_x = 'percent'
    #For each sampler model
    for model in models:
        #Get subset of sample for a given sampler
        m = res[res["Sampler_type"] == model]
        #Plot the results
        plt.plot(m.loc[:,col_x], m.loc[:,col_y],label = model)
    plt.legend()
    plt.show()

