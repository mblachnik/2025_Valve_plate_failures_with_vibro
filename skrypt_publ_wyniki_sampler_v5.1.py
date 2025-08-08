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
# pm = preproc.StandardScaler()
# smote = SMOTE()


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


def plot_data(df, cols, y=None, yp=None):
    for col in cols:
        if y is None:
            idx = np.ones((df.shape[0],), dtype=bool)
        else:
            idx = y!=yp
        x = np.arange(idx.shape[0])
        y = df.loc[idx,col]

        # plt.figure()
        plt.plot(x[idx], y,'.')
        y = df.loc[~idx, col]
        plt.plot(x[~idx], y,'.')
        plt.title(col)
        # plt.show()
        

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
    acc=me.accuracy_score(y_true=y, y_pred=yp)
    print(acc)
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

    return yp, ypp, acc

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
    # pm = preproc.StandardScaler()
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
                    "attr set":mdl['attr set'],
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

def wyniki_dict (models, dataUT1_list, dataUT2, dataUT3, col_y, th_list):
    # f_csv = pd.read_csv(gs_file_csv)   
    # pm = preproc.StandardScaler()

    X = pd.DataFrame()
    y = pd.DataFrame()

    #*********************************************************
    #   WYNIKI !!!
    #*********************************************************
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder_name = './Wyniki_v5 '+time_stamp
    os.mkdir(folder_name)
    
    results_UT23 = []
    for data_name, dataUT1 in dataUT1_list:
        for mdl in models:
            cols_x = mdl['cols']
            number = mdl['number']
            name = mdl['name']
            model = mdl['best']
            sampler = mdl['sampler']
            gs_stdbacc = mdl['std_bacc']
            gs_std_f1macro= mdl['std_f1_macro']
            
            # params = mdl['params']
            gs_refit = mdl['refit']
            # attr_set = mdl["attr set"]
            X_UT1, y_UT1 = prepareData(dataUT1, cols_x, col_y)
            # X_UT1 = pm.fit_transform(X_UT1)
            # X_UT1 = pm.fit_transform(X_UT1)
            # X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)
        
            X_UT2, y_UT2 = prepareData(dataUT2,cols_x,col_y)
            # X_UT2 = pm.transform(X_UT2)
            X_UT3, y_UT3 = prepareData(dataUT3,cols_x,col_y)
            # X_UT3 = pm.transform(X_UT3)
            x_list = []
            x_list.append(('UT2',X_UT2,y_UT2))
            x_list.append(('UT3',X_UT3,y_UT3))
            for xname, X, y in x_list:
                print(f'{number}_{name}_{data_name}_{xname}')
                # print(f"i={i}, j={j}")
                # print(i)
                # print(x_list[i][0])
                # model = mdl
                print(f'Model fitting...')
                model.fit(X_UT1,y_UT1)
                for th in th_list:
                # print(i[1][1])
                # print([i][0])
                # X = i[1][1]
                # y = i[1][2]
                # for j, name, score, params, sigma, model_n, a, b, c in models_list: 
                # for j, name, score, params, sigma, model_n, clmns, smot, bacc  in models_csv: 
                    # name=str(mdl.best_estimator_._final_estimator).split('(')[0]
                    # params=str(mdl.best_params_)

                    print(f'Evaluating feat. import.')
                    evaluateFeatureImportances(model, X, y, np.array(cols_x), threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
                    print(f'Eval. model')
                    yp, ypp, acc = evaluateModel(model, X, y, cols_x ,threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
                    # yp = model.predict(X, y)
                    # ypp= model.predict_proba(X, y)
                    results_UT23.append({
                        "model name":f'{number}_{name}',
                        # "model name":str(mdl.estimator)
                        # "attr set":attr_set,
                        "acc":me.accuracy_score(y_true=y, y_pred=yp),
                        "bal_acc":me.balanced_accuracy_score(y_true=y, y_pred=yp),
                        "f1_macro":me.f1_score(y_true=y, y_pred=yp, average='macro'),
                        "recall_1":me.recall_score(y_true=y, y_pred=yp),                   
                        "auc":roc_auc_score(y, ypp[:,1]),
                        "report":me.classification_report(y_true=y, y_pred=yp),
                        # "params":params,
                        # "params":params,
                        "sampler":sampler,
                        "data":xname,
                        "threshold":th,
                        "model":model,
                        "gs_refit":gs_refit,                      
                        "data_UT1": data_name,
                        "cols":cols_x,
                        "gs_std_bacc":gs_stdbacc,
                        "gs_std_f1_macro":gs_std_f1macro,
                        # "lag":lag,
                        # "mean":res.cv_results_['mean_test_score'][res.best_index_],
                    })
    # print(str(np.random.randn(1)))
    filename = folder_name+'/wyniki25_' + time_stamp + '.csv'
    res_df = pd.DataFrame(results_UT23)
# results_df = pd.DataFrame(results)
    res_df.to_csv(filename, sep=';')
    plt.close('all')
    # return results_UT23
#%% Ładowanie danych v5
data_UT1_100 = pd.read_csv('./data_UT1_v5_100.csv')
data_UT1_050 = pd.read_csv('./data_UT1_v5_050.csv')
# data_UT1_048 = pd.read_csv('./data_UT1_v5_048.csv')
data_UT1_025 = pd.read_csv('./data_UT1_v5_025.csv')
data_UT1_010 = pd.read_csv('./data_UT1_v5_010.csv')
data_UT1_005 = pd.read_csv('./data_UT1_v5_005.csv')
data_UT1_001 = pd.read_csv('./data_UT1_v5_001.csv')
data_UT2 = pd.read_csv('./data_UT2_v5.csv')
data_UT3 = pd.read_csv('./data_UT3_v5.csv')

data_UT1_list = [
    # ('data_UT1_100', data_UT1_100),
    # ('data_UT1_050', data_UT1_050),
    # ('data_UT1_048', data_UT1_048),
    # ('data_UT1_025', data_UT1_025),
    # ('data_UT1_010', data_UT1_010),
    # ('data_UT1_005', data_UT1_005),
    ('data_UT1_001', data_UT1_001)]
#%% Ładowanie pliku grid

# fnames_list = ['grid_search_results_dic_v4_istotne.pickl', 
#                'grid_search_results_dic_v4_ist+wibro.pickl',
#                'grid_search_results_dic_v4_wibro.pickl']
# models=[]
# fname = 'grid_search_results_dic_v4_all_sampler_2025-07-26_204348.pickl'
# fname = 'grid_search_results_v4_sampler_ist+wibro2025-07-26_204348.csv'
# fname2 = './grid_search_results_dic_v5_all_sampler_2025-07-29_153415.pickl'
# grid_search_results_dic_v5_all_sampler_2025-07-29_153415
# fname2 = './grid_search_results_dic_v5_all_sampler_2025-08-04_204001.pickl'
fname2 = './grid_search_results_dic_v5_all_sampler_2025-08-07_180105.pickl' # dane 1%

# metryki = [0,1,6,7,12,13] # metryki dla bacc i f1_macro
# models_csv = pd.DataFrame()
# for fname in fnames_list:
# models_csv = pd.read_csv(fname, sep=';').values.tolist()
# with open(fname,"br") as f: models_dict = pickle.load(f)

with open(fname2,"br") as f: models_dic = pickle.load(f)

models=[]
for n, model in enumerate(models_dic):
    # if len(resdic_all[met]['cols'])==8: attr='istotne'
    # if len(resdic_all[met]['cols'])==11: attr='ist+wibro'
    # if len(resdic_all[met]['cols'])==3: attr='wibro'
    # if model['sampler'] == 'pusta dupa':
        models.append(model    
            |{
                # "attr set":attr,
                "number":n,
                "name":'MLP',
                # "model": model,
                # "params":params,
                # "cols":mdl['cols']
                    })

#%%
# wybrane = [2,5,8,11,34,45]

# models_csv2 = [model for model in models_csv if ('Random' in model[1]) or ('MLP' in model[1])]
# wybrane = [models_csv2[i][0] for i,pos  in enumerate(models_csv2)]
# models_pckl2 = [models_pckl[i] for i in wybrane]   
# results = []


# th_list=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
th_list=[0.5]
# wyniki(models_pckl, models_csv, data_UT1, data_UT2, data_UT3, cols_list[0], col_y, th_list=th_list)
# for model in models:
wyniki_dict(models, data_UT1_list, data_UT2, data_UT3, col_y, th_list=th_list)

# filename = folder_name+'/wyniki25_' + time_stamp + '.csv'
# results_df = pd.DataFrame(results)
# results_df.to_csv(filename, sep=';')
# plt.close('all')    

# wyniki_csv(models_csv, data_UT1, data_UT2, data_UT3, cols_x, col_y, th_list=th_list)

# th_list=[0.5]
# wyniki(models_pckl, models_csv, data_UT1, data_UT2, data_UT3, cols_list[1], col_y, th_list=th_list)
# wyniki([models_pckl[20], models_pckl[26]], [models_csv[20], models_csv[26]], data_UT1, data_UT2, data_UT3, cols_list[1], col_y, th_list=th_list)

# plt.plot(model.loss_curve_)

# models_csv[25, 26]

# resdic_all_df = pd.DataFrame(resdic_all)
# resdic_all_df.to_csv('resdic_all_v4.csv', sep=';')

# models_df = pd.DataFrame(models)
# models_df.to_csv('models_v4.csv', sep=';')

#%% Rysuj wyniki
folder = '_wyniki'
model = models[6]['best']
cols_x = models[6]['cols']

X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y)
# X_UT1 = pm.fit_transform(X_UT1)
model.fit(X_UT1, y_UT1)
    
X_UT2, y_UT2 = prepareData(data_UT2, cols_x ,col_y)
# X_UT2 = pm.transform(X_UT2)
# cols_x_corr = cols_x
# if do_pca:
#     # PCA - temperatura
#     corr = X_UT2[:, ~attrs]
#     un_corr = pca.transform(corr)
#     X_UT2 = np.hstack((X_UT2[:, attrs], un_corr))

yp_UT2 =evaluateModel(model,X_UT2, y_UT2, cols_x, 0.85, folder, 'MLP', 'UT2')

# evaluateFeatureImportances(model, X_UT2, y_UT2, np.array(cols_x_corr))
#%%
%matplotlib qt
plt.close('all')
# for model in [models[2],models[3],models[5]]:
for model in models:
    cols_x = model['cols']
    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y)
    X_UT1 = pm.fit_transform(X_UT1)
    model['best'].fit(X_UT1, y_UT1)
    X_UT2, y_UT2 = prepareData(data_UT2, cols_x ,col_y)
    X_UT2 = pm.transform(X_UT2)
    yp_UT2,ypp,acc  =evaluateModel(model['best'],X_UT2, y_UT2, cols_x, 0.5, folder, 'MLP', 'UT2')
    # acc=me.accuracy_score(y_UT2, yp_UT2[0])
    plt.figure(f'Błędy UT2 {model["number"]} {acc:.3f}',figsize=(10,6))
    plt.title(f'Model {model["number"]} {model["sampler"]}\nacc = {acc:.4f}\n\n')
    for n in range(0, len(cols_x)):
        sizex = int(np.sqrt(len(cols_x)))
        sizey = len(cols_x)//sizex+1
        plt.subplot(sizex,sizey,n+1)
        plt.title(cols_x[n])
        # plot_data(data_UT2.loc[:,cols_x], cols_x, y_UT2, yp_UT2[0])
        # for col in cols_x:
        if y_UT2 is None:
            idx = np.ones((data_UT2.shape[0],), dtype=bool)
        else:
            idx = y_UT2!=yp_UT2
        x = np.arange(idx.shape[0])
        y = data_UT2.loc[idx,cols_x[n]]
        plt.plot(x[idx], y,'.', color='red')
        y = data_UT2.loc[~idx, cols_x[n]]
        plt.plot(x[~idx], y,'.', alpha=0.3, color='green')
        plt.title(cols_x[n])
        # plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()    

#%%
X_UT3, y_UT3 = prepareData(data_UT3,cols_x,col_y)
X_UT3 = pm.transform(X_UT3)
# if do_pca:
#     #PCA - temperatura
#     corr = X_UT3[:, ~attrs]
#     un_corr = pca.transform(corr)
#     X_UT3 = np.hstack((X_UT3[:, attrs], un_corr))

yp_UT3 = evaluateModel(model,X_UT3, y_UT3, cols_x, 0.85, folder, 'MLP', 'UT2')
# evaluateFeatureImportances(model, X_UT3, y_UT3, np.array(cols_x_corr))


#plot_data(data_UT3.loc[:,cols_x])

# plot_data(data_UT2.loc[:,cols_x],['Sensor 2'],y_UT2,yp_UT2[0])
# plot_data(data_UT3.loc[:,cols_x],['Flow - output'],y_UT3,yp_UT3[0])
plt.close('all')
