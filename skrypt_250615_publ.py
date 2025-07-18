import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.ensemble as ens
import sklearn.metrics as me
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from external import RotationForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing as preproc
from sklearn.neighbors import KNeighborsClassifier
import pickle
import datetime
import time
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

from sklearn.tree import plot_tree

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier


#%matplotlib qt

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

def plot_data(df, cols, y=None, yp=None):
    for col in cols:

        if y is None:
            idx = np.ones((df.shape[0],), dtype=bool)
        else:
            idx = y!=yp
        x = np.arange(idx.shape[0])
        y = df.loc[idx,col]

        plt.figure()
        plt.plot(x[idx], y,'.')
        y = df.loc[~idx, col]
        plt.plot(x[~idx], y,'.')
        plt.title(col)
        plt.show()


def addAutoregression(X,lag=2):
    Xls = []
    new_cols = []
    idx = np.ones((X.shape[0]-lag,),dtype=bool)
    for j in range(lag + 1):
        if j == 0:
            Xl = X.iloc[lag - j:, :]
        else:
            Xl = X.iloc[lag - j:-j, :]
        Xl.columns = [col + f"-{j}" for col in Xl.columns]
        Xls.append(Xl)
        if j>0:
            idx &= (Xls[0].index - Xls[j].index)==pd.Timedelta( seconds=j)
        new_cols += list(Xl.columns)
    Xls = [X.reset_index(drop=True) for X in Xls]
    newX = pd.concat(Xls, axis=1, ignore_index=True)
    newX.columns = new_cols
    newX = newX.loc[idx,:]
    return newX, idx

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
        model, X, y, n_repeats=10, random_state=42, n_jobs=2
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
    X_UT1 = pm.fit_transform(X_UT1)
    X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)

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
                name=str(mdl.best_estimator_._final_estimator).split('(')[0]
                params=str(mdl.best_params_)
                print(name,', th=',th)
                # print(f"i={i}, j={j}")
                # print(i)
                # print(x_list[i][0])
                model = mdl.best_estimator_
                print(f'Model fitting...')
                model.fit(X_UT1,y_UT1)
                print(f'Evaluating feat. import.')
                evaluateFeatureImportances(model._final_estimator, X, y, np.array(cols_x), threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
                print(f'Eval. model')
                yp, ypp = evaluateModel(model._final_estimator, X, y, cols_x ,threshold=th, folder=folder_name, m_name=str(number)+'_'+name, data_name=xname)
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
# %% Load data files
fNames = ['dane_OT.csv',
          'dane_UT1.csv',
          'dane_UT2.csv',
          'dane_UT3.csv']
dfs = []
temps = pd.DataFrame()
fleak = pd.DataFrame()
torque = pd.DataFrame()

for f in fNames:

    df = pd.read_csv('data250505/' + f)
    temps = pd.concat([temps,pd.DataFrame(([(df["Temperature - suction line"].min(), df["Temperature - suction line"].max())]))])
    fleak = pd.concat([fleak,pd.DataFrame(([(df["Flow - leak line"].min(), df["Flow - leak line"].max())]))])  # print(f, ' temp min - ', df['Temperature - suction line'].min())
    torque = pd.concat([torque,pd.DataFrame(([(df["Applied torque"].min(), df["Applied torque"].max())]))])  # print(f, ' temp min - ', df['Temperature - suction line'].min())

    df = df.reset_index(drop=True)
    dfs.append(df)

    
#%% Make dfs[x]
# dfs=[]
# dfs.append(data_UT1)
# dfs.append(data_UT2)
# dfs.append(data_UT3)
           
for i in range(len(dfs)):
    dfs[i] = dfs[i][
            (dfs[i]['Applied torque']>19)
            & (dfs[i]['Applied torque']<221)
            # & (dfs[i]['Temperature - suction line']> temps[0].max()) 
            # & (dfs[i]['Temperature - suction line']< temps[1].min()) 
            ]
    dfs[i].dropna(inplace = True)
    dfs[i].drop_duplicates(inplace=True)
    dfs[i] = dfs[i].reset_index(drop=True)
# data_UT1 = pd.DataFrame()
# data_UT1 = dfs[0]
# data_UT2 = dfs[1]
# data_UT3 = dfs[2]

#%% Choose columns
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
         # 'Pressure - leak line', #2 0
          'Temperature - leak line', #3 1
          # 'Pressure - output', #4 2
          'Temperature - suction line', #5 3
          'Temperature - output', #6 4
         #  'Flow - leak line', #7 5
           'Flow - output',#8 6
              # 'Sensor 1',#9 7
         #  #   'Sensor 2',#10 8
         #  #   'Sensor 3',#11 9
           'Temp. diff', #12 10
            # 'hPower',
            # 'Fleak_mul_Pout'
         ]

cols_x_vib_imp = [
         # 'Applied torque',
         # 'Pressure - leak line', #2 0
         # 'Temperature - leak line', #3 1
          'Pressure - output', #4 2
         # 'Temperature - suction line', #5 3
          'Temperature - output', #6 4
           'Flow - leak line', #7 5
         #  'Flow - output',#8 6
         #    'Sensor 1',#9 7
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

col_y =  "stan"

cols_list = []
cols_list.append(cols_x_imp)
cols_list.append(cols_x_vib_imp)


# #%%
# trening = połowa OT do treningu (ok. 3500-4000) + sztuczne (upsampling.smote)

# test = reszta OT (ok. 3500)

# miara dokł F1-makro albo dokładn. zbalans (balanced accuracy)

# ustawić w Gridsearch`u optymalizację balanced acc.


#%%

train_data_factor=0.6
    
df0train=dfs[0].sample(round(train_data_factor*dfs[0].shape[0]))

df0test=dfs[0].drop(df0train.index)


# plt.figure('train')
# plt.plot(df0train['Temperature - suction line'],',')
# plt.figure('test')
# plt.plot(df0test['Temperature - suction line'],',')

plt.figure('dfs_0')
plt.plot(dfs[0]['Temperature - suction line'],'.')
plt.plot(dfs[0]['stan'],',')
plt.plot(dfs[0]['Pressure - output'],'.')

# plt.figure()
# plt.plot(df0train['Applied torque'],',')

plt.figure('X_UT1')
plt.plot(X_UT1[:,1],'.',label='p_out')
plt.plot(X_UT1[:,5],'.',label='f_out')
plt.plot(y_UT1,'.')
plt.legend()

plt.figure()
# plt.plot(dfs[0]['stan'],',')
plt.plot(dfs[0]['Flow - output'],',')

 
#%%

data_UT1 = pd.concat([
            df0train,
            dfs[1],
        ], axis=0)

data_UT2 = pd.concat([
            df0test,
            dfs[2]
        ],axis=0)

data_UT3 = pd.concat([
            df0test,
            dfs[3]
        ], axis=0)

data_UT1.reset_index(inplace=True)
data_UT2.reset_index(inplace=True)
data_UT3.reset_index(inplace=True)

print('\nData UT1: \n',data_UT1['stan'].value_counts(sort=False))

print('\nData UT2: \n', data_UT2['stan'].value_counts(sort=False))

print('\nData UT3: \n', data_UT3['stan'].value_counts(sort=False))

# plt.plot(data_UT1['Applied torque'].reset_index(),'.')
# plt.figure()
# plt.plot(data_UT1.index, data_UT1['Applied torque'],'.')
# plt.plot(data_UT1.index, data_UT1['stan']*200,'.')

# %%
n_splits=10
do_cv = False
# do_cv = True
# do_grid_search = False
do_grid_search = True
# smote = SMOTE()

if do_cv:
    lag = 0
    cv = ms.StratifiedKFold(n_splits=n_splits)
    model = RandomForestClassifier(n_estimators=100, max_depth=7, max_features=0.3, n_jobs=6)
    # model = MLPClassifier(hidden_layer_sizes=(8,), learning_rate_init=0.01, max_iter=100)
    # model = MLPClassifier(early_stopping = True, hidden_layer_sizes=(40,20), learning_rate_init=0.001, max_iter=800)
    # model = ens.GradientBoostingClassifier(n_estimators = 300, max_depth = 7, learning_rate = 0.2)
    # model = MLPClassifier(
    #         # early_stopping = True,
    #         hidden_layer_sizes=(10,6), learning_rate_init=0.001, max_iter=100)
    # model=RotationForestClassifier(n_estimators=100,max_depth=7, max_features=0.3,n_jobs=6)
    pm = preproc.StandardScaler()

    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
    X_UT1 = pm.fit_transform(X_UT1)
    res = ms.cross_val_score(model, X_UT1, y_UT1, cv=cv, n_jobs=n_splits)
    print(res)
    print( f"{np.mean(res)} +- {np.std(res)}")

if do_grid_search:
    print("Start GridSearch")
    cv = ms.StratifiedKFold(n_splits=n_splits)
    models = [
        ('RandomForest',
              RandomForestClassifier(n_jobs=6),
              {"randomforestclassifier__n_estimators":[50,100,200,300],
                "randomforestclassifier__max_depth":[9,15,50],
                "randomforestclassifier__max_features":[0.3,0.5,0.6]}
            ),
        ('RotationForest',
              RotationForestClassifier(n_jobs=6),
              {"rotationforestclassifier__n_estimators": [50, 100, 200, 300],
              "rotationforestclassifier__max_depth": [9, 12, 37],
              "rotationforestclassifier__n_features_per_subset": [2,3,4]}
          ),

        ('GradientBoostedTrees',
            GradientBoostingClassifier(),
            {"gradientboostingclassifier__n_estimators": [50, 100, 200, 300],
              "gradientboostingclassifier__max_depth": [7, 9, 12, 29],
              "gradientboostingclassifier__learning_rate": [0.05, 0.1, 0.2]}
          ),
        ('kNN',
          KNeighborsClassifier(n_jobs=6,),
          {
              "kneighborsclassifier__n_neighbors":[1,3,5,7,9,11,15,21,29,51,71,101,201],
              "kneighborsclassifier__weights":["uniform", "distance"]
          }
          ),
        ('MLP',
          MLPClassifier(),
          {"mlpclassifier__hidden_layer_sizes": [(4,),(8,),(12,),(10,6),(10,4),(40,20),
              (10,),(30,),(50,),(100,),
              (100,10)],
          "mlpclassifier__max_iter": [200,500,1000],
          "mlpclassifier__learning_rate_init": [0.01,0.001,0.0001]}
          ),

        ('XGBoost',
          XGBClassifier(),
          {'xgbclassifier__n_estimators': [100, 300, 500],
          'xgbclassifier__max_depth': [3, 5, 7,9,11],
          'xgbclassifier__learning_rate': [0.1, 0.2, 0.4, 0.6],
          'xgbclassifier__subsample': [0.8, 1.0, 1.5, 2.0],
          'xgbclassifier__colsample_bytree': [0.8, 1.0, 1.5]}
          )


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
        #   {"mlpclassifier__hidden_layer_sizes": [(4,),(8,)],
        #   "mlpclassifier__max_iter": [200],
        #   "mlpclassifier__learning_rate_init": [0.01]}
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
    pm = preproc.StandardScaler()

    scorings=['balanced_accuracy','f1_macro']
    res_bin = []
    res_dic = []

    for cols_gs in cols_list:
        X_UT1, y_UT1 = prepareData(data_UT1, cols_gs, col_y)
        # X_UT1 = pm.fit_transform(X_UT1)
        # X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)
        # model_gs = make_pipeline(pm,smote,model)
        for name,model,param in models:
            for scoring in scorings:
                for smoted in [False, True]:
                    print(f" {model} started")
                    if smoted:
                        model_gs = make_pipeline(pm,smote,model)
                    else:
                        model_gs = make_pipeline(pm,model)
                    res = ms.GridSearchCV(model_gs,param,cv=cv,n_jobs=10, scoring=scoring)
                    res.fit(X_UT1,y_UT1)
                    res_dic.append({
                        "model":name,
                        "score":res.best_score_,
                        "params":str(res.best_params_),
                        # "lag":lag,
                        # "mean":res.cv_results_['mean_test_score'][res.best_index_],
                        "std":res.cv_results_['std_test_score'][res.best_index_],
                        "best":res.best_estimator_,
                        "cols":cols_gs,
                        "Scoring":scoring,
                        "SMOTED":str(smoted),
                    })
                    print(f" {model} finished")
                    print(f"    {res.best_score_}")
                    # print(f"    {res.best_params_}")
                    # print(f"    {lag}")
                    print("-------------------------------------------")
                    res_bin.append(res)
                # res.best_estimator_
    df = pd.DataFrame(res_dic)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    filename_csv = 'grid_search_results_' + time_stamp + '.csv'
    df.to_csv(filename_csv, sep=';')

    filename_pickl = 'grid_search_results_' + time_stamp + '.pickl'
    with open(filename_pickl,"bw") as f:
        pickle.dump(res_bin, f)
        
#%% Wczytanie danych z ustawionym limitem temperatury na ssaniu
data_UT1 = pd.read_csv('data_UT1_t-limit.csv')
data_UT2 = pd.read_csv('data_UT2_t-limit.csv')
data_UT3 = pd.read_csv('data_UT3_t-limit.csv')
#%% Wczytanie danych bez limitu temperatury na ssaniu
data_UT1 = pd.read_csv('data250707/data_UT1.csv')
data_UT2 = pd.read_csv('data250707/data_UT2.csv')
data_UT3 = pd.read_csv('data250707/data_UT3.csv')
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

print(df0test['Temperature - suction line'].min(),
      df0train['Temperature - suction line'].min(),
      dfs[0]['Temperature - suction line'].min()
      )

#%%
smote = SMOTE()
time_stamp='2025-06-20_194958'     

filename_csv = 'grid_search_results_' + time_stamp + '.csv'
filename_pickl = 'grid_search_results_' + time_stamp + '.pickl'
models_csv = pd.read_csv(filename_csv, sep=';').values.tolist()
with open(filename_pickl,"br") as f_pckl:
    models_pckl = pickle.load(f_pckl)
#%%    
th_list=[0.3,0.2,0.1]
wyniki(models_pckl, models_csv, data_UT1, data_UT2, data_UT3, cols_x_imp, col_y, th_list=th_list)

th_list=[0.5]
wyniki(models_pckl, models_csv, data_UT1, data_UT2, data_UT3, cols_x_vib_only, col_y, th_list=th_list)
# wyniki([models_pckl[20], models_pckl[26]], [models_csv[20], models_csv[26]], data_UT1, data_UT2, data_UT3, cols_list[1], col_y, th_list=th_list)

# plt.plot(model.loss_curve_)

# models_csv[25, 26]


#%%
with open(filename_pickl,"br") as f:
    models_pckl=pickle.load(f)

for idx, mdl in enumerate(models_pckl):
   print(f'{idx}_ {mdl.best_estimator_._final_estimator}') 

# wyniki (gs_file_pckl, gs_file_csv, dataUT1, dataUT2, dataUT3, cols_x, col_y):
models_csv = pd.read_csv(filename_csv, sep=';').values.tolist()
# models_csv[0].insert(9,models_pckl[0])
for i, dane in enumerate(models_csv):
    models_csv[i].insert(len(models_csv[0]), models_pckl[i])

with open(filename_pickl,"br") as f_pckl:
    models_pckl=pickle.load(f_pckl)
# f_csv = pd.read_csv(gs_file_csv)   
X_UT1, y_UT1 = prepareData(dataUT1, cols_x, col_y)
X_UT1 = pm.fit_transform(X_UT1)
X_UT1, y_UT1 = smote.fit_resample(X_UT1, y_UT1)

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
for th in [0.5, 0.4, 0.3, 0.2, 0.1]:
    for xname, X, y in x_list:
        # print(i[1][1])
        # print([i][0])
        # X = i[1][1]
        # y = i[1][2]
        # for j, name, score, params, sigma, model_n, a, b, c in models_list: 
        # for j, name, score, params, sigma, model_n, clmns, smot, bacc  in models_csv: 
        for mdl in models_pckl:
            name=str(mdl.best_estimator_._final_estimator).split('(')[0]
            params=str(mdl.best_params_)
            print(name,', th=',th)
            # print(f"i={i}, j={j}")
            # print(i)
            # print(x_list[i][0])
            model = mdl.best_estimator_
            model.fit(X_UT1,y_UT1)


#%%
mdl=eval(res_dic[0]['model']+'('+res_dic[0]['params']+')')
mdl.fit(X_UT1, y_UT1)

mdl2=res_bin[5].best_estimator_
mdl2.fit(X_UT1, y_UT1)

#%%
th=0.5
X_UT2, y_UT2 = prepareData(data_UT2,cols_x,col_y)
X_UT2 = pm.transform(X_UT2)
cols_x_corr = cols_x
# if do_pca:
#     # PCA - temperatura
#     corr = X_UT2[:, ~attrs]
#     un_corr = pca.transform(corr)
#     X_UT2 = np.hstack((X_UT2[:, attrs], un_corr))
print("*************************\nTest UT2")
  
yp_UT2, ypp =evaluateModel(model,X_UT2, y_UT2, [#(0,'Pressure - leak line'),
                                           #(2,'Pressure - output'), #4
                                            (4,'Temperature - output'), #6
                                            (5,'Flow - leak line')], #7
                                           # (7,'Temp. diff'), #7
                                            #(6, 'Flow - output')],
                      threshold = th)

evaluateFeatureImportances(model, X_UT2, y_UT2, np.array(cols_x_corr), '\nLearn UT1 - FI -UT2')

test_auc = roc_auc_score(y_UT2, ypp[:,1])
# print("Wynik AUC dla RF na zbiorze testowym UT2:", test_auc)
print("Wynik AUC dla MLP na zbiorze testowym UT2:", test_auc)

plt.figure()
plt.plot(ypp[:,1],'.')  

#%% tree importances

importances = model.feature_importances_
#std = np.std([tree.feature_importances_ for tree in dec_tree.estimators_], axis=0)
# elapsed_time = time.time() - start_time

#print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

dec_tree_importances = pd.Series(importances, index=cols_x)

fig, ax = plt.subplots()
dec_tree_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI - DT")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

#%% Wykres drzewa
wynik = confusion_matrix(y_UT2, yp_UT2)
pltwynik = ConfusionMatrixDisplay(wynik, display_labels=model.classes_)
pltwynik.plot()
plt.figure('DT max_depth: '+str(5)+str(datetime.time()), clear=True)
plot_tree(model, feature_names = cols_x, fontsize = 10)
#%% Rysuj Feature importances
#start_time = time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
#elapsed_time = time.time() - start_time

#print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
#forest_importances.plot.bar(yerr=std, ax=ax)
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI - RF")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


#%%    
    ypp = model.predict_proba(X_UT2)
    proba_res = pd.DataFrame(ypp)
    
    # yp=ypp
    plt.figure("Proba UT2")
    plt.plot(proba_res[0],'bo')
    plt.plot(proba_res[1],'rx') 
#%%
X_UT3, y_UT3 = prepareData(data_UT3,cols_x,col_y,lag)
X_UT3 = pm.transform(X_UT3)
print("*************************\nTest UT3")

yp_UT3 = evaluateModel(model, X_UT3, y_UT3, [#(2,'Pressure - output'), #4
                                            (4,'Temperature - output'), #6
                                            (5,'Flow - leak line')], #7
                                          #  (6, 'Flow - output')],
                                              threshold = th)

# yp = model.predict(X_UT3)
# print(me.accuracy_score(y_UT3, yp))
# print(me.classification_report(y_UT3, y_pred=yp))
# cm = confusion_matrix(y_UT3, yp)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, )
# disp.plot()
# plt.title(f'{model}')
# plt.show()
# idCor = y_UT3 == yp

# for j,(i,col) in enumerate(columns):
#     plt.figure(col)
#     x = np.linspace(0, idCor.shape[0], idCor.shape[0])
#     y = X[:, i]
    
#     plt.plot(x[idCor], y[idCor], '.b')
#     plt.plot(x[~idCor], y[~idCor], '.r')
#     plt.ylabel(columns[j])
#     # plt.title('UT2')
#     plt.show()

evaluateFeatureImportances(model, X_UT3, y_UT3, np.array(cols_x_corr), '\nLearn UT1 - FI-UT3')

#%%
    lag=0
    pm = preproc.StandardScaler()
    cols_x_corr = cols_x

    # model = MLPClassifier(hidden_layer_sizes=(40,20), learning_rate_init=0.01, max_iter=500, random_state=42)
    model = ens.GradientBoostingClassifier(n_estimators = 200, max_depth = 9, learning_rate = 0.1)
    model2 = model
    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
    X_UT1 = pm.fit_transform(X_UT1)

    model.fit(X_UT1, y_UT1)

    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
    X_UT1 = pm.transform(X_UT1)
    yp=model.predict(X_UT1)
    print('\n', accuracy_score(y_UT1, yp))
    print(me.classification_report(y_UT1,yp))
    evaluateFeatureImportances(model, X_UT1, y_UT1, np.array(cols_x_corr), '\nLearn UT1 - FI-UT1')

    X_UT2, y_UT2 = prepareData(data_UT2, cols_x, col_y, lag)
    X_UT2 = pm.transform(X_UT2)
    yp=model.predict(X_UT2)
    print('\n', accuracy_score(y_UT2, yp))
    print(me.classification_report(y_UT2,yp))
    evaluateFeatureImportances(model, X_UT2, y_UT2, np.array(cols_x_corr), '\nLearn UT1 - FI-UT2')
    
    X_UT3, y_UT3 = prepareData(data_UT3, cols_x, col_y, lag)
    X_UT3 = pm.transform(X_UT3)
    yp=model.predict(X_UT3)
    print('\n', accuracy_score(y_UT3, yp))
    print(me.classification_report(y_UT3,yp))
    evaluateFeatureImportances(model, X_UT3, y_UT3, np.array(cols_x_corr), '\nLearn UT1 - FI-UT3')

#%%
    model = ens.GradientBoostingClassifier(n_estimators = 200, max_depth = 9, learning_rate = 0.1)
    model2 = model
    # model = MLPClassifier(hidden_layer_sizes=(40,20), learning_rate_init=0.001, max_iter=500, random_state=42)
    X_UT2, y_UT2 = prepareData(data_UT2, cols_x, col_y, lag)
    X_UT2 = pm.fit_transform(X_UT2)

    model.fit(X_UT2, y_UT2)

    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
    X_UT1 = pm.transform(X_UT1)
    yp=model.predict(X_UT1)
    print('\n', accuracy_score(y_UT1, yp))
    print(me.classification_report(y_UT1,yp))
    evaluateFeatureImportances(model, X_UT1, y_UT1, np.array(cols_x_corr), '\nFit UT2 - FI-UT1')

    X_UT2, y_UT2 = prepareData(data_UT2, cols_x, col_y, lag)
    X_UT2 = pm.transform(X_UT2)
    yp=model.predict(X_UT2)
    print('\n', accuracy_score(y_UT2, yp))
    print(me.classification_report(y_UT2,yp))
    evaluateFeatureImportances(model, X_UT2, y_UT2, np.array(cols_x_corr), '\nFit UT2 - FI-UT2')
    
    X_UT3, y_UT3 = prepareData(data_UT3, cols_x, col_y, lag)
    X_UT3 = pm.transform(X_UT3)
    yp=model.predict(X_UT3)
    print('\n', accuracy_score(y_UT3, yp))
    print(me.classification_report(y_UT3,yp))
    evaluateFeatureImportances(model, X_UT3, y_UT3, np.array(cols_x_corr), '\nFit UT2 - FI-UT3')
# %%
    model = ens.GradientBoostingClassifier(n_estimators = 200, max_depth = 9, learning_rate = 0.1)
    model2 = model

    # model = MLPClassifier(hidden_layer_sizes=(40,20), learning_rate_init=0.001, max_iter=500, random_state=42)
    X_UT3, y_UT3 = prepareData(data_UT3, cols_x, col_y, lag)
    X_UT3 = pm.fit_transform(X_UT3)

    model.fit(X_UT3, y_UT3)

    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
    X_UT1 = pm.transform(X_UT1)
    yp=model.predict(X_UT1)
    print('\n', accuracy_score(y_UT1, yp))
    print(me.classification_report(y_UT1,yp))
    evaluateFeatureImportances(model, X_UT1, y_UT1, np.array(cols_x_corr), '\nFit UT3 - FI-UT1')

    X_UT2, y_UT2 = prepareData(data_UT2, cols_x, col_y, lag)
    X_UT2 = pm.transform(X_UT2)
    yp=model.predict(X_UT2)
    print('\n', accuracy_score(y_UT2, yp))
    print(me.classification_report(y_UT2,yp))
    evaluateFeatureImportances(model, X_UT2, y_UT2, np.array(cols_x_corr), '\nFit UT3 - FI-UT2')
    
    X_UT3, y_UT3 = prepareData(data_UT3, cols_x, col_y, lag)
    X_UT3 = pm.transform(X_UT3)
    yp=model.predict(X_UT3)
    print('\n', accuracy_score(y_UT3, yp))
    print(me.classification_report(y_UT3,yp))
    evaluateFeatureImportances(model, X_UT3, y_UT3, np.array(cols_x_corr), '\nFit UT3 - FI-UT3')

#%%
X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y)
X_UT1 = pm.transform(X_UT1)
yp=model.predict(X_UT1)
print('\n', accuracy_score(y_UT1, yp))
print(me.classification_report(y_UT1,yp))
evaluateFeatureImportances(model, X_UT1, y_UT1, np.array(cols_x_corr), '\nFit UT1plus - FI-UT1')
#%%
X_UT1, y_UT1 = prepareData(data_UT1plus, cols_x, col_y, lag)
X_UT1 = pm.transform(X_UT1)
yp=model.predict(X_UT1)
print('\n', accuracy_score(y_UT1, yp))
print(me.classification_report(y_UT1,yp))
evaluateFeatureImportances(model, X_UT1, y_UT1, np.array(cols_x_corr), '\nFit UT1plus - FI-UT1plus')


#%%
evaluateFeatureImportances(model, X_UT1, y_UT1, np.array(cols_x_corr), '\nLearn UT1 - FI-UT1')
evaluateFeatureImportances(model, X_UT1, y_UT1, np.array(cols_x_corr), '\nLearn UT1 - FI-UT1plus')

    ypp = model.predict_proba(X_UT3)
    proba_res = pd.DataFrame(ypp)
    
    plt.figure("Proba UT3")
    plt.plot(proba_res[0],'bo')
    plt.plot(proba_res[1],'rx')


    #%%
plt.close('all')

data_UT1.to_csv('data_UT1.csv')
data_UT2.to_csv('data_UT2.csv')
data_UT3.to_csv('data_UT3.csv')

modelsdf.to_csv('models.csv', sep=';')
