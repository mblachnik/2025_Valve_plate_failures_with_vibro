import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.model_selection as ms
import sklearn.ensemble as ens
import sklearn.metrics as me
import sklearn.preprocessing as preproc

import pickle
import datetime
import time
import os
import copy

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc

from xgboost import XGBClassifier

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

from external import RotationForestClassifier


# %matplotlib qt
pm = preproc.StandardScaler()
smote = SMOTE()


feat_map = {
             'Pressure - leak line':"$P_{leak}$",
             'Temperature - leak line':"$T_{leak}$",
             'Pressure - output':"$P_{out}$",
             'Temperature - suction line':"$ T_{suct.}$",
             'Temperature - output':"$T_{out}$",
             'Flow - leak line':"$F_{leak}$",
             'Flow - output':"$F_{out}$",
             'Temp. diff':"$T_{diff}$",
             'Sensor 1':"$Vib_{1}$",
             'Sensor 2':"$Vib_{2}$",
             'Sensor 3':"$Vib_{3}$",
             'hPower':"$P_{hyd}$",
             'Fleak_mul_Pout':'$F_{leak}mulP_{out}$',
             'T':'T'
}


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

col_y = "stan"

# cols_list = []
# # cols_list.append(cols_x)
# # cols_list.append(cols_x_vib)
# cols_list.append(cols_x_imp)
# cols_list.append(cols_x_imp_vib)
# cols_list.append(cols_x_vib_only)


def prepareData(data, cols_x, col_y):
    X = data.loc[:, cols_x]
    y = data.loc[:, col_y]
    return X,y


#%% Ładowanie danych v4
data_UT1 = pd.read_csv('./data_UT1_v5.csv')
data_UT2 = pd.read_csv('./data_UT2_v5.csv')
data_UT3 = pd.read_csv('./data_UT3_v5.csv')

# %% Grid Search
n_splits=10

print("Start GridSearch")
cv = ms.StratifiedKFold(n_splits=n_splits)
models = [
    # ('RandomForest',
    #       RandomForestClassifier(n_jobs=6),
    #       {"randomforestclassifier__n_estimators":[50,100,200,300],
    #         "randomforestclassifier__max_depth":[9,15,50],
    #         "randomforestclassifier__max_features":[0.3,0.5,0.6]}
    #     ),
    # ('RotationForest',
    #       RotationForestClassifier(n_jobs=6),
    #       {"rotationforestclassifier__n_estimators": [50, 100, 200, 300],
    #       "rotationforestclassifier__max_depth": [9, 12, 37],
    #       "rotationforestclassifier__n_features_per_subset": [2,3,4]}
    #   ),

    # ('GradientBoostedTrees',
    #     GradientBoostingClassifier(),
    #     {"gradientboostingclassifier__n_estimators": [50, 100, 200, 300],
    #       "gradientboostingclassifier__max_depth": [7, 9, 12, 29],
    #       "gradientboostingclassifier__learning_rate": [0.05, 0.1, 0.2]}
    #   ),
    # ('kNN',
    #   KNeighborsClassifier(n_jobs=6,),
    #   {
    #       "kneighborsclassifier__n_neighbors":[1,3,5,7,9,11,15,21,29,51,71,101,201],
    #       "kneighborsclassifier__weights":["uniform", "distance"]
    #   }
    # #   ),
    ('MLP',
      MLPClassifier(),
      {"mlpclassifier__hidden_layer_sizes": [(4,),(8,),(12,),(10,6),(10,4),(40,20),(60,40),(70,15),
          (10,),(30,),(50,),(100,),
          (100,10)],
      "mlpclassifier__max_iter": [200,500,1000],
      "mlpclassifier__learning_rate_init": [0.01,0.001,0.0001],
       }
       ),
    # ('Log. regr',
    #  LogisticRegressionCV(scoring='f1_macro'),
    #  {}
    #  ),

    # ('XGBoost',
    #   XGBClassifier(),
    #   {'xgbclassifier__n_estimators': [100, 300, 500],
    #   'xgbclassifier__max_depth': [3, 5, 7,9,11],
    #   'xgbclassifier__learning_rate': [0.1, 0.2, 0.4, 0.6],
    #   'xgbclassifier__subsample': [0.8, 1.0, 1.5, 2.0],
    #   'xgbclassifier__colsample_bytree': [0.8, 1.0, 1.5]}
    #   )


#*************** Test models ************************
    # ('RandomForest',
    #       RandomForestClassifier(n_jobs=6),
    #       {"randomforestclassifier__n_estimators":[50],
    #         "randomforestclassifier__max_depth":[9,15],
    #         # "randomforestclassifier__max_features":[0.3]
    #         }
    #     ),
    # # ('RotationForest',
    # #       RotationForestClassifier(n_jobs=6),
    # #       {"rotationforestclassifier__n_estimators": [50],
    # #       "rotationforestclassifier__max_depth": [3,9],
    #       # "rotationforestclassifier__n_features_per_subset": [2]}
    #   # ),

    # # ('GradientBoostedTrees',
    # #     GradientBoostingClassifier(),
    # #     {"gradientboostingclassifier__n_estimators": [50, 100],
    # #       "gradientboostingclassifier__max_depth": [7],
    # #       "gradientboostingclassifier__learning_rate": [0.05]}
    # #   ),
    # ('kNN',
    #   KNeighborsClassifier(n_jobs=6,),
    #   {
    #       "kneighborsclassifier__n_neighbors":[1,3,5,7],
    #       # "kneighborsclassifier__weights":["uniform", "distance"]
    #   }
    #   ),
    # ('MLP',
    #   MLPClassifier(),
    #   {"mlpclassifier__hidden_layer_sizes": [(4,)],
    #   "mlpclassifier__max_iter": [100,200],
    #   "mlpclassifier__learning_rate_init": [0.1]}
    #   ),

    # ('XGBoost',
    #   XGBClassifier(),
    #   {'xgbclassifier__n_estimators': [100],
    #   'xgbclassifier__max_depth': [3],
    #   # 'xgbclassifier__learning_rate': [0.1, 0.2],
    #   # 'xgbclassifier__subsample': [0.8],
    #   # 'xgbclassifier__colsample_bytree': [0.8]
    #   }
    #   )
    
    ]

scorings=['balanced_accuracy','f1_macro', 'accuracy',
          'recall', 'roc_auc', 'average_precision']

# scorings=['balanced_accuracy']
res_bin = []
res_dic = []

# cols_list = [cols_x_imp, cols_x_imp_vib, cols_x_vib_only]
cols_list = [
            ['istotne', cols_x_imp],
            ['ist+wibro',cols_x_imp_vib],
            ['wibro', cols_x_vib_only]
            ]
# cols_dict = {
#             'istotne':cols_x_imp,
#             'ist+wibro':cols_x_imp_vib,
#             'wibro':cols_x_vib_only
#              }

for etykieta, cols_gs in cols_list:
    X_UT1, y_UT1 = prepareData(data_UT1, cols_gs, col_y)
    # X_UT1 = pm.fit_transform(X_UT1)
    # X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)
    # model_gs = make_pipeline(pm,smote,model)
    for name,model,param in models:
        param_gs=param.copy()
        # for smoted in [False, True]:
        for smoted in [False]:
            for scoring in scorings:
                print(f" {model}, SMOTE: {smoted} started")
                if smoted:
                    model_gs = make_pipeline(pm,smote,model)
                    # param["smote__k_neighbors"] = [2,3,5,7,9]
                    param_gs["smote__k_neighbors"] = [2,3,5,7,9,11]
                else:
                    model_gs = make_pipeline(pm,model)
                    # param = param_gs
                res = ms.GridSearchCV(model_gs,param_gs,
                                      cv=cv,n_jobs=-1,
                                      scoring=scorings,
                                      refit="f1_macro",return_train_score=False)
                res.fit(X_UT1,y_UT1)
                res_dic.append({
                    # "cechy":cols_list[nr][0],
                    "model":name,
                    "score":res.best_score_,
                    "params":str(res.best_params_),
                    # "lag":lag,
                    # "mean":res.cv_results_['mean_test_score'][res.best_index_],
                    # "std":res.cv_results_['std_test_score'][res.best_index_],
                    # "cv_res":res.cv_results_,
                    "best":res.best_estimator_,
                    "cols":cols_gs,
                    "Scoring":scoring,
                    # "SMOTED":str(smoted),
                } | res.cv_results_)
                print(f" {model} finished")
                # print(f"    {res.best_score_}")
                # print(f"    {res.best_params_}")
                # print(f"    {lag}")
                print("-------------------------------------------")
                res_bin.append(res)
            # res.best_estimator_
    res_dic_df = pd.DataFrame(res_dic)
    # cvresdf=pd.DataFrame(res.cv_results_)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    filename_csv = 'grid_search_results_v4_' + etykieta + time_stamp + '.csv'
    res_dic_df.to_csv(filename_csv, sep=';')
    
    filename_pickl = 'grid_search_results_v4_' + etykieta + time_stamp + '.pickl'
    with open(filename_pickl,"bw") as f:
        pickle.dump(res_bin, f)
    
    filename_pickl = 'grid_search_results_dic_v4_' + etykieta + time_stamp + '.pickl'
    with open(filename_pickl,"bw") as f:
        pickle.dump(res_dic_df, f)

    filename_pickl = 'grid_search_results_dic_v4_all' + time_stamp + '.pickl'
    with open(filename_pickl,"bw") as f:
        pickle.dump(res_dic, f)
        

