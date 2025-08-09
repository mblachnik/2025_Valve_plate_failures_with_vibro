#Additional required packages:
#imbalanced-learn
#scikit-learn
#pandas
#matplotlib
#seaborn
#umap-learn

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import get_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib
from my_permutation_importance import permutation_importance


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def swap_labels(y):
    """
    Labels are swapped so that instead of 0,1 we have 1,0 class 0 is 1 and class 1 is 0
    :param y:
    :return:
    """
    y = np.copy(y)
    y = np.abs(y-1)
    return y

def correct_labels(yp):
    """
    Labels are converted from -1,1 to 1,0 such that 1 is converted to 0, and -1 to 1
    :param y:
    :return:
    """
    yp = np.copy(yp)
    yp[yp == 1] = 0
    yp[yp == -1] = 1
    return yp

def correct_labels_df(yp,th=0):
    """
    Labels are converted based on given threshold, if value is greater or equal to threshold then 0,
    if value is less than threshold then 1,
    """
    yp = np.copy(yp)
    yp[yp >= th] = 0
    yp[yp < th] = 1
    return yp

def correct_labels_y(y):
    """
    Labels are converted from 0,1 to -1,1 such that 1 is converted to -1, and 0 to 1
    :param y:
    :return:
    """
    y = np.copy(y)
    y[y == 1] = -1
    y[y == 0] = 1
    return y

def eval_permutations(model, X:pd.DataFrame,y,cols_groups: list):
    """
    Function used for feature importance evaluation
    :param model: prediction model - yp=model.predict(X) must be present
    :param X: input data - must be pandas DataFrame with proper column names
    :param y: labels
    :param cols_groups: list of column names or list of lists of column names. In the second case not a single column
    but all columns in the list are permuted. For example in case of iris with columns ["a1,"a2","a3","a4"] wi can set
    [["a1],["a2","a3"],["a4","a1"]] then once permutations are evaluated for a1, then for a2 and a3 for which the two
    columns are permuted simmultanousely, and finally [a4 and a1] are permuted simmultanousely
    :return:
    """
    r = permutation_importance(model, X, y,
                               n_repeats=30,
                               random_state=random_state,
                               scoring=get_scorer("f1_macro"),
                               cols_groups=cols_groups)
    sorted_importances_idx = r.importances_mean.argsort()
    cols = []
    for cg in cols_groups:
        cols.append(", ".join(cg))
    importances = pd.DataFrame(
        r.importances[sorted_importances_idx].T,
        columns=cols
        #X.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.show()

percent = 1
detect_failure_states = False #When True then labels are replaced from 0-normal, 1-failure to 0-failure, 1-normal -
# When True model is trained on failure state and normal state is an anomaly
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
#%% Reading and preparing data
df1 = pd.read_csv(f"data/mb/data1_{percent * 100}.csv")
# df1 = pd.read_csv(f"data/mb/data1_2024.csv")
X1 = df1.loc[:, cols_x_imp_vib]
y1 = df1.loc[:, 'stan'].to_numpy()

df2 = pd.read_csv(f"data/mb/dataA_2.csv")
# df2 = pd.read_csv(f"data/mb/data2_2024.csv")
X2 = df2.loc[:, cols_x_imp_vib]
y2 = df2.loc[:, 'stan'].to_numpy()

df3 = pd.read_csv(f"data/mb/dataA_3.csv")
# df3 = pd.read_csv(f"data/mb/data3_2024.csv")
X3 = df3.loc[:, cols_x_imp_vib]
y3 = df3.loc[:, 'stan'].to_numpy()


# %% Preparing model
random_state = 21

model = make_pipeline(
     StandardScaler(),
            LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=4)
            #IsolationForest(n_estimators=300, random_state=random_state, n_jobs=4)
            #OneClassSVM()
        )

#%% Train the model only on X1 for class y1==0.
#By default it trains on normal state, but if detect_failure_states==True then  labels are swapped and model is train on failure1 class
if detect_failure_states: #
    y1 = swap_labels(y1)  # To check if failure states are similar
    y2 = swap_labels(y2)  # To check if failure states are similar
    y3 = swap_labels(y3) #To check if failure states are similar

Xtr = X1.loc[y1==0, :]
model.fit(Xtr)

#%% Make prediction
y1p = model.predict(X1)
y2p = model.predict(X2)
y3p = model.predict(X3)
y1b = correct_labels_y(y1) #Correct labels form 0,1 to 1,-1
y2b = correct_labels_y(y2) #Correct labels form 0,1 to 1,-1
y3b = correct_labels_y(y3) #Correct labels form 0,1 to 1,-1

#Calculate performance
metrics = ['f1_macro', 'balanced_accuracy']
text = "; Metric ;  Performanc X1 ; Performanc X1 ; Performanc X1" * len(metrics)
text += "\n"
for scorer_name in metrics:
    scorer = get_scorer(scorer_name)
    sc1 = scorer._score_func(y1b, y1p)
    sc2 = scorer._score_func(y2b, y2p)
    sc3 = scorer._score_func(y3b, y3p)
    text += f" ; {scorer_name} ;  {sc1 * 100:.2f} ; {sc2 * 100:.2f} ; {sc3 * 100:.2f}"
print(text)
#%% Evaluation feature importanc - feature groups
eval_permutations(model, X3,y3b,cols_groups=[
                ['Pressure - leak line',  # 2 0
                 'Pressure - output'],  # 4 2
                ['Temperature - leak line',  # 3 1
                'Temperature - suction line',  # 5 3
                'Temperature - output'],  # 6 4
                ['Flow - leak line',  # 7 5
                'Flow - output'],  # 8 6
                ['Sensor 1',  # 9 7
                'Sensor 2',  # 10 8
                'Sensor 3'],  # 11 9
                ['Temp. diff'],  # 12 10
])

#%% Evaluation feature importanc - individual features
eval_permutations(model, X3,y3b,cols_groups=[
                ['Pressure - leak line'],  # 2 0
                 ['Pressure - output'],  # 4 2
                ['Temperature - leak line'],  # 3 1
                ['Temperature - suction line'],  # 5 3
                ['Temperature - output'],  # 6 4
                ['Flow - leak line'],  # 7 5
                ['Flow - output'],  # 8 6
                ['Sensor 1'],  # 9 7
                ['Sensor 2'],  # 10 8
                ['Sensor 3'],  # 11 9
                ['Temp. diff'],  # 12 10
])
#%% Wizualizacja danych
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import umap
def tsne_classification_pipeline(X, y, class_names=None):
    """
    Complete t-SNE visualization pipeline for classification data

    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target labels (n_samples,)
    class_names: optional list of class names
    """

    # Step 1: Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Optional PCA preprocessing for high-dimensional data
    if X.shape[1] > 50:
        print(f"High dimensional data ({X.shape[1]} features). Applying PCA first...")
        pca = PCA(n_components=50)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # Step 3: Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,  # Typical values: 5-50
        max_iter=1000,  # Increase for better convergence
        learning_rate=200,  # Auto or 200 is often good
        early_exaggeration=12,  # Default is 12
        metric='euclidean'  # Can also try 'manhattan', 'cosine'
    )

    X_tsne = tsne.fit_transform(X_scaled)

    # Step 4: Visualization
    plt.figure(figsize=(12, 10))

    # Use seaborn for better colors if many classes
    if len(np.unique(y)) <= 10:
        palette = sns.color_palette("tab10", len(np.unique(y)))
    else:
        palette = sns.color_palette("husl", len(np.unique(y)))

    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        label_name = class_names[i] if class_names else f'Class {class_label}'
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=[palette[i]], label=label_name, alpha=0.7, s=50)

    plt.title('t-SNE Visualization of Classification Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X_tsne

def pca_classification_pipeline(X, y, class_names=None):
    """
    Complete PCA visualization pipeline for classification data

    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target labels (n_samples,)
    class_names: optional list of class names
    """

    # Step 1: Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply PCA

    print(f"High dimensional data ({X.shape[1]} features). Applying PCA ...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # Step 4: Visualization
    plt.figure(figsize=(12, 10))

    # Use seaborn for better colors if many classes
    if len(np.unique(y)) <= 10:
        palette = sns.color_palette("tab10", len(np.unique(y)))
    else:
        palette = sns.color_palette("husl", len(np.unique(y)))

    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        label_name = class_names[i] if class_names else f'Class {class_label}'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[palette[i]], label=label_name, alpha=0.7, s=50)

    plt.title('PCA Visualization of Classification Data')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X_pca

def umap_classification_pipeline(X, y, class_names=None):
    """
    Complete UMAP visualization pipeline for classification data

    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target labels (n_samples,)
    class_names: optional list of class names
    """

    # Step 1: Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply UMAP

    print(f"High dimensional data ({X.shape[1]} features). Applying UMAP...")
    umap_viz = umap.UMAP(n_components=2,
                         n_neighbors=30,  # Number of neighbors (5-50)
                         min_dist=0.1,  # Minimum distance between points
                         metric='euclidean',  # Distance metric
                         random_state=42,
                         )
    X_umap = umap_viz.fit_transform(X_scaled)

    # Step 4: Visualization
    plt.figure(figsize=(12, 10))

    # Use seaborn for better colors if many classes
    if len(np.unique(y)) <= 10:
        palette = sns.color_palette("tab10", len(np.unique(y)))
    else:
        palette = sns.color_palette("husl", len(np.unique(y)))

    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        label_name = class_names[i] if class_names else f'Class {class_label}'
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                    c=[palette[i]], label=label_name, alpha=0.7, s=50)

    plt.title('UMAP Visualization of Classification Data')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X_umap

#Data concatenation
X = pd.concat([X1, X2, X3])
#Labels concatenation, each label has new unique value
y = np.hstack([y1, y2*2, y3*3])
#%% Visualization T-SNE
tsne_classification_pipeline(X, y,["normal","f1","f2","f3"])


#%% Visualization PCA - meaningless results
pca_classification_pipeline(X, y,["normal","f1","f2","f3"])

#%% Visualization UMAP
umap_classification_pipeline(X, y,["normal","f1","f2","f3"])

