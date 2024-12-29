from pyclustering.cluster.clique import clique, clique_visualizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
import optuna
from functools import partial
import numpy as np
from main import read_data
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(true_labels, pred_labels):
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    cost_matrix = np.zeros((len(unique_true), len(unique_pred)))

    for i, t in enumerate(unique_true):
        for j, p in enumerate(unique_pred):
            cost_matrix[i, j] = -np.sum((true_labels == t) & (pred_labels == p))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_matches = -cost_matrix[row_ind, col_ind].sum()

    accuracy = total_matches / len(true_labels)
    return accuracy

def evaluate_clustering(true_labels, pred_labels, data):
    label_encoder = LabelEncoder()
    encoded_true_labels = label_encoder.fit_transform(true_labels)
    accuracy = cluster_accuracy(encoded_true_labels, pred_labels)
    ari = adjusted_rand_score(encoded_true_labels, pred_labels)
    silhouette = silhouette_score(data, pred_labels)
    return accuracy, ari, silhouette

# Objective function
def objective(trial, train_data, features):
    start_time = time.time()
    
    variance_threshold = trial.suggest_float('variance_threshold', 0, 0.2)
    n_components = trial.suggest_float('n_components', 0.7, 1.0)
    grid_size = trial.suggest_int('grid_size', 5, 30)  # CLIQUE's amount_intervals
    tau = trial.suggest_float('tau', 0.01, 0.1)  # CLIQUE's density_threshold

    # Data preprocessing
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    X_train = pp.fit_transform(train_data[features])
    true_labels = train_data['activity']
    
    # CLIQUE clustering
    clique_instance = clique(data=X_train, amount_intervals=grid_size, density_threshold=tau)
    clique_instance.process()
    clusters = clique_instance.get_clusters()
    
    # Assign labels
    pred_labels = np.zeros(X_train.shape[0], dtype=int) - 1
    for cluster_id, cluster_points in enumerate(clusters):
        pred_labels[cluster_points] = cluster_id
    
    # Evaluate clustering
    accuracy, ari, silhouette = evaluate_clustering(true_labels, pred_labels, X_train)
    
    end_time = time.time()
    print(f"Trial finished in {end_time - start_time:.4f} seconds")
    
    # Return silhouette score as optimization goal
    return silhouette

# Main Program
if __name__ == "__main__":
    train_data, test_data, features = read_data()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, train_data=train_data, features=features), n_trials=40)
    
    # Best Parameters
    best_params = study.best_params
    best_value = study.best_value
    
    variance_threshold = best_params['variance_threshold']
    n_components = best_params['n_components']
    grid_size = best_params['grid_size']
    tau = best_params['tau']
    
    # Preprocess data
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    
    X_train = pp.fit_transform(train_data[features])
    true_train_labels = train_data['activity']
    X_test = pp.transform(test_data[features])
    true_test_labels = test_data['activity']
    
    # Train CLIQUE
    clique_instance = clique(X_train, intervals=grid_size, threshold=tau)
    clique_instance.process()
    train_clusters = clique_instance.get_clusters()
    
    pred_train_labels = np.zeros(X_train.shape[0], dtype=int) - 1
    for cluster_id, cluster_points in enumerate(train_clusters):
        pred_train_labels[cluster_points] = cluster_id
    
    # Evaluate
    accuracy_train, ari_train, silhouette_train = evaluate_clustering(
        true_train_labels, pred_train_labels, X_train)
    
    print(f"Best Parameters: {best_params}")
    print("Train Set Results:")
    print(f"  Accuracy: {round(accuracy_train, 4)}")
    print(f"  Adjusted Rand Score: {round(ari_train, 4)}")
    print(f"  Silhouette Score: {round(silhouette_train, 4)}")