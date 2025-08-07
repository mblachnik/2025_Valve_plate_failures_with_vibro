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

# from external import RotationForestClassifier


# %matplotlib qt
pm = preproc.StandardScaler()

smote = SMOTE()
randomOS = RandomOverSampler()
randomUS = RandomUnderSampler()
nearm = NearMiss()
enn = EditedNearestNeighbours()
smoteenn = SMOTEENN()


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

cols_list = []
# cols_list.append(cols_x)
# cols_list.append(cols_x_vib)
cols_list.append(cols_x_imp)
cols_list.append(cols_x_imp_vib)
cols_list.append(cols_x_vib_only)


def prepareData(data, cols_x, col_y):
    X = data.loc[:, cols_x]
    y = data.loc[:, col_y]
    return X,y

def evaluateModel(model, X,y, columns, threshold, folder, m_name, data_name):
    # yp = np.array(int)
    yp = model.predict(X)
    ypp = model.predict_proba(X)
    if (threshold != 0.5):
        for i in range(len(ypp)):
            if ypp[i,1] > threshold:
                yp[i]=1
            else:
                yp[i]=0
    
    print(me.accuracy_score(y_true=y, y_pred=yp))
    print(me.classification_report(y_true=y, y_pred=yp))
    cm = confusion_matrix(y, yp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, )
    fig, ax = plt.subplots(figsize=(8,6),layout='constrained')
    disp.plot(ax=ax)
    ax.set_title(f'Confusion matrix - {m_name}\n {data_name}, th={threshold}\n')
    # plt.show()
    fig_name=str(np.random.randn(1))
    plt.savefig(folder+'/'+m_name+data_name+str(threshold)+fig_name+'.jpg')
    plt.savefig(folder+'/'+m_name+data_name+str(threshold)+fig_name+'.pdf')
    # plt.savefig(folder+'/'+fig_name+'.pdf')
    plt.close()
    idCor = y == yp

    # for j,(i,col) in enumerate(columns):
    #     plt.figure(col)
    #     x = np.linspace(0, idCor.shape[0], idCor.shape[0])
    #     y = X[:, i]
        
    #     plt.plot(x[idCor], y[idCor], '.b')
    #     plt.plot(x[~idCor], y[~idCor], '.r')
    #     plt.ylabel(columns[j])
    #     # plt.title('UT2')
    #     # plt.show()

    return yp, ypp

def evaluateFeatureImportances(model,X,y, columns, threshold, folder, m_name, data_name):
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=5
        )
    sorted_importances_idx = result.importances_mean.argsort()
    print(sorted_importances_idx)

    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=[feat_map[c] for c in columns[sorted_importances_idx]],
    )
    ax = importances.plot.box(vert=False, whis=20)
    # ax = importances.plot.barh()
    # ax.set_title(f"Permutation Importances - {model} - {name}")
    ax.set_title(f'Permutation importances - {m_name}\n {data_name}, th={threshold}\n')
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    fig_name=str(np.random.randn(1))
    plt.savefig(folder+'/'+m_name+data_name+str(threshold)+fig_name+'.jpg')
    plt.savefig(folder+'/'+m_name+data_name+str(threshold)+fig_name+'.pdf')
    # plt.savefig(folder+'/'+fig_name+'.pdf')
    plt.close()
    # plt.show()

def wyniki (models_pckl, models_csv, dataUT1, dataUT2, dataUT3, cols_x, col_y, th_list):
    # f_csv = pd.read_csv(gs_file_csv)   
    pm = preproc.StandardScaler()
    X_UT1, y_UT1 = prepareData(dataUT1, cols_x, col_y)
    # X_UT1 = pm.fit_transform(X_UT1)
    _ = pm.fit_transform(X_UT1)
    # X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)

    X_UT2, y_UT2 = prepareData(dataUT2,cols_x,col_y)
    X_UT2 = pm.transform(X_UT2)
    X_UT3, y_UT3 = prepareData(dataUT3,cols_x,col_y)
    X_UT3 = pm.transform(X_UT3)
    x_list = []
    x_list.append(('UT2',X_UT2,y_UT2))
    x_list.append(('UT3',X_UT3,y_UT3))

    X = pd.DataFrame()
    y = pd.DataFrame()

    #*********************************************************
    #   WYNIKI !!!
    #*********************************************************
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = './Wyniki '+time_stamp
    os.mkdir(folder_name)
    results_UT23 = []
    for th in th_list:
        for xname, X, y in x_list:
            # print(i[1][1])
            # print([i][0])
            # X = i[1][1]
            # y = i[1][2]
            # for j, name, score, params, sigma, model_n, a, b, c in models_list: 
            # for j, name, score, params, sigma, model_n, clmns, smot, bacc  in models_csv: 
            for number, mdl in enumerate(models_pckl):
                # name=str(mdl.best_estimator_._final_estimator).split('(')[0]
                params=str(mdl.best_params_)
                print(name,', th=',th)
                # print(f"i={i}, j={j}")
                # print(i)
                # print(x_list[i][0])
                model = mdl.best_estimator_
                print(f'Model fitting...')
                model.fit(X_UT1,y_UT1)
                print(f'Evaluating feat. import.')
                evaluateFeatureImportances(model, X, y, np.array(cols_x), threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
                print(f'Eval. model')
                yp, ypp = evaluateModel(model, X, y, cols_x ,threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
                # yp = model.predict(X, y)
                # ypp= model.predict_proba(X, y)
                results_UT23.append({
                    "model name":f'{number}_{name}',
                    # "model name":str(mdl.estimator)
                    "acc":me.accuracy_score(y_true=y, y_pred=yp),
                    "bal_acc":me.balanced_accuracy_score(y_true=y, y_pred=yp),
                    "f1_macro":me.f1_score(y_true=y, y_pred=yp, average='macro'),
                    "recall_1":me.recall_score(y_true=y, y_pred=yp),                   
                    "report":me.classification_report(y_true=y, y_pred=yp),
                    "auc":roc_auc_score(y, ypp[:,1]),
                    # "params":params,
                    "params":params,
                    "data":xname,
                    "model":model,
                    "threshold":th,
                    "cols":cols_x
                    # "lag":lag,
                    # "mean":res.cv_results_['mean_test_score'][res.best_index_],
                })
    # print(str(np.random.randn(1)))
    res_df = pd.DataFrame(results_UT23)
    filename = folder_name+'/wyniki25_' + time_stamp + '.csv'
    res_df.to_csv(filename, sep=';')
    plt.close('all')

def wyniki_dict (models, dataUT1, dataUT2, dataUT3, cols_x, col_y, th_list):
    # f_csv = pd.read_csv(gs_file_csv)   
    pm = preproc.StandardScaler()
    X_UT1, y_UT1 = prepareData(dataUT1, cols_x, col_y)
    # X_UT1 = pm.fit_transform(X_UT1)
    X_UT1 = pm.fit_transform(X_UT1)
    # X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)

    X_UT2, y_UT2 = prepareData(dataUT2,cols_x,col_y)
    X_UT2 = pm.transform(X_UT2)
    X_UT3, y_UT3 = prepareData(dataUT3,cols_x,col_y)
    X_UT3 = pm.transform(X_UT3)
    x_list = []
    x_list.append(('UT2',X_UT2,y_UT2))
    x_list.append(('UT3',X_UT3,y_UT3))

    X = pd.DataFrame()
    y = pd.DataFrame()

    #*********************************************************
    #   WYNIKI !!!
    #*********************************************************
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = './Wyniki '+time_stamp
    os.mkdir(folder_name)
    results_UT23 = []
    for th in th_list:
        for xname, X, y in x_list:
            # print(i[1][1])
            # print([i][0])
            # X = i[1][1]
            # y = i[1][2]
            # for j, name, score, params, sigma, model_n, a, b, c in models_list: 
            # for j, name, score, params, sigma, model_n, clmns, smot, bacc  in models_csv: 
            # for number, name, mdl, params in models:
                number = models['number']
                name = models['name']
                model = models['model']
                params = models['params']
                # name=str(mdl.best_estimator_._final_estimator).split('(')[0]
                # params=str(mdl.best_params_)
                print(name,', th=',th)
                # print(f"i={i}, j={j}")
                # print(i)
                # print(x_list[i][0])
                # model = mdl
                print(f'Model fitting...')
                model.fit(X_UT1,y_UT1)
                print(f'Evaluating feat. import.')
                evaluateFeatureImportances(model, X, y, np.array(cols_x), threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
                print(f'Eval. model')
                yp, ypp = evaluateModel(model, X, y, cols_x ,threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
                # yp = model.predict(X, y)
                # ypp= model.predict_proba(X, y)
                results_UT23.append({
                    "model name":f'{number}_{name}',
                    # "model name":str(mdl.estimator)
                    "acc":me.accuracy_score(y_true=y, y_pred=yp),
                    "bal_acc":me.balanced_accuracy_score(y_true=y, y_pred=yp),
                    "f1_macro":me.f1_score(y_true=y, y_pred=yp, average='macro'),
                    "recall_1":me.recall_score(y_true=y, y_pred=yp),                   
                    "report":me.classification_report(y_true=y, y_pred=yp),
                    "auc":roc_auc_score(y, ypp[:,1]),
                    # "params":params,
                    "params":params,
                    "data":xname,
                    "model":model,
                    "threshold":th,
                    "cols":cols_x
                    # "lag":lag,
                    # "mean":res.cv_results_['mean_test_score'][res.best_index_],
                })
    # print(str(np.random.randn(1)))
    res_df = pd.DataFrame(results_UT23)
    filename = folder_name+'/wyniki25_' + time_stamp + '.csv'
    res_df.to_csv(filename, sep=';')
    plt.close('all')    
#%% Ładowanie danych v4
data_UT1 = pd.read_csv('./data_UT1_v4.csv')
data_UT2 = pd.read_csv('./data_UT2_v4.csv')
data_UT3 = pd.read_csv('./data_UT3_v4.csv')

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

balance={
        "smote": SMOTE(),
        "RandomOS": RandomOverSampler(),
        "RandomUS": RandomUnderSampler(),
        }
# scorings=['balanced_accuracy']
res_bin = []
res_dic = []

# cols_list = [cols_x_imp, cols_x_imp_vib, cols_x_vib_only]
cols_list = [[cols_x_imp, cols_x_imp_vib, cols_x_vib_only],
             ['istotne', 'ist+wibro', 'wibro']]
# cols_dict = {
#             'istotne':cols_x_imp,
#             'ist+wibro':cols_x_imp_vib,
#             'wibro':cols_x_vib_only
#              }

for wer, etykieta, cols_gs in cols_list:
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
        

#%%
with open('res_dic.pickl',"br") as f:
    res_dic = pickle.load(f)
    
res_dic_2=copy.deepcopy(res_dic)
for i,e in enumerate(res_dic):
    for k in res_dic[i]:
        if "mean_test" in k:
            # print(f'{k}: {np.mean(res_dic[i][k])}')
            res_dic_2[i][str('_mean_'+k)] = np.mean(res_dic[i][k])
            res_dic_2[i][str('_min_'+k)] = np.min(res_dic[i][k])
            
        # if "std_test" in k:
        #     # print(f'{k}: {np.mean(res_dic_2[i][k])}')
        #     res_dic_2[i][str('_mean_'+k)] = np.mean(res_dic[i][k])
        if ("split" in k) and ("test" in k):
            # print(f'{k}: {np.mean(res_dic[i][k])}')
            # res_dic_2[i][str('_mean_'+k)] = np.mean(res_dic[i][k])
            res_dic_2[i][str('_min_'+k)] = np.min(res_dic[i][k])
#%%
# res_df=[]
# for i, ble in enumerate(res_dic):
#     hg=[]
keys = ["model",
"score",
# "params",
# "lag":lag,
# "mean":res.cv_results_['mean_test_score'][res.best_index_],
# "std":res.cv_results_['std_test_score'][res.best_index_],
# "cv_res":res.cv_results_,
"best",
"cols",
# "Scoring":scoring,
"SMOTED"]
for k in res_dic[0]:
    if "split" in k:
        keys.append(k)
#%%
for i, eee in enumerate(res_dic):
    # for k in keys:
    #     del res_df[k]
    nowy_slownik = {
        klucz: wartosc
        for klucz, wartosc in res_dic[i].items()
        if klucz not in keys # Ważne: Sprawdzamy, czy klucz faktycznie istnieje w oryginalnym słowniku
        }
    plik = 'result_'+str(i)+'_'+res_dic[i]['model']+'.csv'
    pd.DataFrame(nowy_slownik).to_csv(plik, sep=';')
    

#%%
print('\nData UT1: \n',data_UT1['stan'].value_counts(sort=False),
      '\n',data_UT1['Temperature - suction line'].min(),
      '\n',data_UT1['Temperature - suction line'].max()
      )
print('\nData UT2: \n', data_UT2['stan'].value_counts(sort=False),
      '\n',data_UT2['Temperature - suction line'].min(),
      '\n',data_UT2['Temperature - suction line'].max()
      )
print('\nData UT3: \n', data_UT3['stan'].value_counts(sort=False),
      '\n',data_UT3['Temperature - suction line'].min(),
      '\n',data_UT3['Temperature - suction line'].max()
      )


#%% Ładowanie modeli z plików
# time_stamp='2025-06-20_194958'     
time_stamp='2025-07-21_061152'

filename_csv = 'grid_search_results_' + time_stamp + '.csv'
filename_pickl = 'grid_search_results_' + time_stamp + '.pickl'
models_csv = pd.read_csv(filename_csv, sep=';').values.tolist()
with open(filename_pickl,"br") as f_pckl:
    models_pckl = pickle.load(f_pckl)
#%% Ładowanie modeli z plików 2

filename_csv = '../res_df.csv'
filename_pickl = '../res_dic.pickl'
models_csv = pd.read_csv(filename_csv, sep=';').values.tolist()
with open(filename_pickl,"br") as f_pckl:
    models_pckl = pickle.load(f_pckl)

rank_f1_macro = True
# rank_f1_macro = False

params = []
models = []
indx_list =[11, 22, 4, 649, 3, 22, 65, 192] # Najlepsze modele MLP GS_rank f1_macro+acc+bacc

for n, mdl in enumerate(models_pckl):
    if rank_f1_macro:
        indx = np.where(mdl['rank_test_f1_macro']==1)[0][0]
        model = mdl['best'],
        params = mdl['params'][indx]
    else:
        indx = indx_list[n]
        params = mdl['params'][indx]
        if mdl['SMOTED']=='True':
            model = make_pipeline(pm, smote, MLPClassifier())
        else:
            model = make_pipeline(pm, MLPClassifier())
            # params = mdl['params'][indx]
            # model.set_params(**params)
        model.set_params(**params)
    models.append({
        "number":n,
        "name":'MLP',
        "model": model,
        "params":params,
        "cols":mdl['cols']
            })
    # models_pckl[n]['best_params_']=(models_pckl[n]['params'][indeks])
    # models_pckl[n]['best_estimator_']=(models_pckl[n]['best'])
    
    # param = models_pckl[0]['best']
#%%
# wybrane = [2,5,8,11,34,45]

# models_csv2 = [model for model in models_csv if ('Random' in model[1]) or ('MLP' in model[1])]
# wybrane = [models_csv2[i][0] for i,pos  in enumerate(models_csv2)]
# models_pckl2 = [models_pckl[i] for i in wybrane]   


th_list=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# wyniki(models_pckl, models_csv, data_UT1, data_UT2, data_UT3, cols_list[0], col_y, th_list=th_list)
for model in models:
    wyniki_dict(model, data_UT1, data_UT2, data_UT3, model['cols'], col_y, th_list=th_list)
# wyniki_csv(models_csv, data_UT1, data_UT2, data_UT3, cols_x, col_y, th_list=th_list)

# th_list=[0.5]
# wyniki(models_pckl, models_csv, data_UT1, data_UT2, data_UT3, cols_list[1], col_y, th_list=th_list)
# wyniki([models_pckl[20], models_pckl[26]], [models_csv[20], models_csv[26]], data_UT1, data_UT2, data_UT3, cols_list[1], col_y, th_list=th_list)

# plt.plot(model.loss_curve_)

# models_csv[25, 26]




    #%%
plt.close('all')

data_UT1.to_csv('data_UT1.csv')
data_UT2.to_csv('data_UT2.csv')
data_UT3.to_csv('data_UT3.csv')

# modelsdf.to_csv('models.csv', sep=';')
