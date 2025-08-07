import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

import sklearn.model_selection as ms
# import sklearn.ensemble as ens
# import sklearn.metrics as me
import sklearn.preprocessing as preproc

import pickle
import datetime
# import time
# import os
import copy

# from sklearn.inspection import permutation_importance
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import plot_tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score, roc_curve, auc

# from xgboost import XGBClassifier

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

# from external import RotationForestClassifier

random_state = 23
# %matplotlib qt
pm = preproc.StandardScaler()

smote = SMOTE(random_state=random_state)
randomOS = RandomOverSampler(random_state=random_state)
randomUS = RandomUnderSampler(random_state=random_state)
nearm = NearMiss()
enn = EditedNearestNeighbours()
smoteenn = SMOTEENN(random_state=random_state)

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


#%% Ładowanie danych v5
data_UT1_048 = pd.read_csv('./data_UT1_v5_048.csv')
data_UT1_025 = pd.read_csv('./data_UT1_v5_025.csv')
data_UT1_010 = pd.read_csv('./data_UT1_v5_010.csv')
data_UT2 = pd.read_csv('./data_UT2_v5.csv')
data_UT3 = pd.read_csv('./data_UT3_v5.csv')

# %% Grid Search
n_splits=10

print("Start GridSearch")
# cv = ms.StratifiedKFold(n_splits=n_splits,random_state=random_state)
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
    # ('MLP',
    #   MLPClassifier(),
    #   {"mlpclassifier__hidden_layer_sizes": [(4,),(8,),(12,),(10,6),(10,4),(40,20),(60,40),(70,15),
    #       (10,),(30,),(50,),(100,),
    #       (100,10)],
    #   "mlpclassifier__max_iter": [200,500,1000],
    #   "mlpclassifier__learning_rate_init": [0.01,0.001,0.0001],
    #    }
    #    ),
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
    ('MLP',
      MLPClassifier(random_state=random_state),
      {"mlpclassifier__hidden_layer_sizes": [(40,20)],
      "mlpclassifier__max_iter": [500],
      "mlpclassifier__learning_rate_init": [0.01]}
      ),

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

samplers = ['smote', 'rand_os', 'rand_us', 'n_miss', 'enn', 'smote_enn', 'pusta dupa']
# samplers = ['rand_us', 'smote_enn']
# samplers = ['pusta dupa']
data_UT1_list = [
    ('data_UT1_048', data_UT1_048),
    ('data_UT1_025', data_UT1_025),
    ('data_UT1_010', data_UT1_010)]
# scorings=['balanced_accuracy']
res_bin = []
res_dic = []
res_cv = []

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
# cols_gs = cols_x_imp_vib


for data_name, data_UT1 in data_UT1_list:
    for etykieta, cols_gs in cols_list:
        X_UT1, y_UT1 = prepareData(data_UT1, cols_gs, col_y)
        # X_UT1 = pm.fit_transform(X_UT1)
        # X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)
        # model_gs = make_pipeline(pm,smote,model)
        for name,model,param in models:
            for sampler in samplers:
                param_gs = copy.deepcopy(param)
                if sampler == 'smote':
                    model_gs = make_pipeline(pm,smote,model)
                    # param["smote__k_neighbors"] = [2,3,5,7,9]
                    param_gs["smote__k_neighbors"] = [2,3,5,7,9,11]
                elif sampler == 'rand_os':
                    model_gs = make_pipeline(pm,randomOS,model)
                elif sampler == 'rand_us':
                    model_gs = make_pipeline(pm,randomUS,model)
                elif sampler == 'n_miss':
                    model_gs = make_pipeline(pm,nearm,model)
                    param_gs["nearmiss__n_neighbors"] = [2,3,5,7,9,11]
                    param_gs["nearmiss__n_neighbors_ver3"] = [2,3,5,7,9,11]
                elif sampler == 'enn':
                    model_gs = make_pipeline(pm,enn,model)
                    param_gs["editednearestneighbours__n_neighbors"] = [2,3,5,7,9,11]
                    # param_gs["smoteenn__enn"] = [EditedNearestNeighbours(n_neighbors=2), EditedNearestNeighbours(n_neighbors=3), 
                    #                                EditedNearestNeighbours(n_neighbors=5), EditedNearestNeighbours(n_neighbors=7),
                    #                                EditedNearestNeighbours(n_neighbors=9), EditedNearestNeighbours(n_neighbors=11),
                    #                                ]
                elif sampler == 'smote_enn':
                    model_gs = make_pipeline(pm,smoteenn,model)
                    # param_gs["smoteenn__smote__k_neighbors"] = [2,3,5,7,9,11]
                    param_gs["smoteenn__smote"] = [SMOTE(k_neighbors=2,random_state=random_state), SMOTE(k_neighbors=3,random_state=random_state), 
                                                    SMOTE(k_neighbors=5,random_state=random_state), SMOTE(k_neighbors=7,random_state=random_state),
                                                    SMOTE(k_neighbors=9,random_state=random_state), SMOTE(k_neighbors=11,random_state=random_state),
                                                    ]
                    param_gs["smoteenn__enn"] = [EditedNearestNeighbours(n_neighbors=2), EditedNearestNeighbours(n_neighbors=3), 
                                                    EditedNearestNeighbours(n_neighbors=5), EditedNearestNeighbours(n_neighbors=7),
                                                    EditedNearestNeighbours(n_neighbors=9), EditedNearestNeighbours(n_neighbors=11),
                                                    ]
                else:
                    model_gs = make_pipeline(pm,model)
                    # param = param_gs
                # for smoted in [False, True]:
                for r_fit in ['f1_macro','balanced_accuracy']:
                    print(f" {model}, sampler:{sampler} refit:{r_fit} started")
                    res = ms.GridSearchCV(model_gs,param_gs,
                                          cv=cv,n_jobs=-1,
                                          scoring=scorings,
                                          refit=r_fit ,return_train_score=False)
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
                        "refit":r_fit,
                        "sampler":sampler,
                        "data_UT1": data_name,
                        "etykieta":etykieta,
                        "std_f1_macro":res.cv_results_['std_test_f1_macro'][0],
                        "std_bacc":res.cv_results_['std_test_balanced_accuracy'][0]
                        # "SMOTED":str(smoted),
                    })
                    res_cv.append(res.cv_results_)
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
filename_csv = 'grid_search_results_v5_sampler_' + time_stamp + '.csv'
res_dic_df.to_csv(filename_csv, sep=';')

filename_pickl = 'grid_search_results_bin_v5_sampler_'+ time_stamp + '.pickl'
with open(filename_pickl,"bw") as f: pickle.dump(res_bin, f)

filename_pickl = 'grid_search_results_dic_v5_sampler_'+ time_stamp + '.pickl'
with open(filename_pickl,"bw") as f:
    pickle.dump(res_dic_df, f)

filename_pickl = 'grid_search_results_dic_v5_all_sampler_' + time_stamp + '.pickl'
with open(filename_pickl,"bw") as f:
    pickle.dump(res_dic, f)
    
filename_pickl = 'grid_search_results_dic_v5_all_cv_res_' + time_stamp + '.pickl'
with open(filename_pickl,"bw") as f:
    pickle.dump(res_cv, f)

for i, result in enumerate(res_cv):
    plik = 'result_v5_'+str(i)+'_'+res_dic[i]['model']+res_dic[i]['best'].steps[1][0]+time_stamp+'.csv'
    pd.DataFrame(result).to_csv(plik, sep=';')
        
