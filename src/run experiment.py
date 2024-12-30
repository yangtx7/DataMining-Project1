import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score, adjusted_rand_score, confusion_matrix, pairwise_distances
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline


import optuna
from main import agglomerative_cluster, compactness, read_data, cluster_accuracy
from functools import partial

def objective(trial, algorithm):
    variance_threshold = trial.suggest_float('variance_threshold', 0, 0.2)
    n_components = trial.suggest_float('n_components', 0.7, 1.0)
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    X_train = pp.fit_transform(train_data[features])
    
    if algorithm == 'kmeans':
        init = trial.suggest_categorical('init', ['k-means++', 'random'])
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, init = init)
        train_clusters = kmeans.fit_predict(train_data_pca)
    elif algorithm == 'agg':
        linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
        agg = AgglomerativeClustering(n_clusters=6, linkage=linkage)

    le = LabelEncoder()
    true_labels_train = le.fit_transform(train_data['activity'])
    accuracy = cluster_accuracy(true_labels_train, pred_labels_train)
    return accuracy

result = pd.read_excel(os.path.join('./result', 'result.xlsx'))
train_data, test_data, features = read_data()
study = optuna.create_study(direction='maximize')


for algorithm in ["kmeans", "agg"]:
    study.optimize(partial(objective, algorithm=algorithm), n_trials=50)
    # print("best_params: ", study.best_params)
    # print("best accuracy: ", study.best_value)
    result = result.append({
        "algorithm": algorithm,
        "best_params": study.best_params,
        "accuracy": round(study.best_value, 2),
        "adjusted_rand_score": 0,
        "compactness": 0,
        "separation": 0,
        "silhouette_score":0
    })
result.to_excel(os.path.join('result', 'result.xlsx'))