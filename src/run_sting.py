import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
import optuna
from functools import partial
import numpy as np
from main import read_data  # 你自己的数据读取函数
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
import seaborn as sns

##############################################
# --------------  STING 实现 ---------------
##############################################

class STING:
    def __init__(self, grid_size=10, density_threshold=0.01):
        self.grid_size = grid_size
        self.density_threshold = density_threshold
        self.cluster_centers_ = None
        self.labels_ = None
        self.active_cells_ = None
        self.cell_idx_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        
        # 计算每个点对应的网格坐标
        cell_indices = []
        for d in range(n_features):
            diff = (self.max_[d] - self.min_[d]) or 1e-9
            indices_d = np.floor((X[:, d] - self.min_[d]) / diff * self.grid_size)
            indices_d = np.clip(indices_d, 0, self.grid_size - 1)
            cell_indices.append(indices_d.astype(int))
        self.cell_idx_ = np.stack(cell_indices, axis=1)
        
        # 统计各网格内点的数量
        cell_counts = {}
        for idx in self.cell_idx_:
            idx_tuple = tuple(idx.tolist())
            cell_counts[idx_tuple] = cell_counts.get(idx_tuple, 0) + 1
        
        # 计算密度
        cell_density = {k: v / n_samples for k, v in cell_counts.items()}
        
        # 筛选活跃网格
        self.active_cells_ = {k for k, density in cell_density.items()
                              if density >= self.density_threshold}
        
        # 连通分量分析
        active_cells_list = list(self.active_cells_)
        visited = set()
        clusters = []
        
        def neighbors(cell):
            offsets = [-1, 0, 1]
            # np.ndindex(*( [3]*len(cell) )) 会产生 (0,0,...,0) 到 (2,2,...,2) 的所有索引组合
            for offset_idx in np.ndindex(*( [3]*len(cell) )):
                # offset_idx[i] ∈ {0,1,2}，对应 offsets[0] = -1, offsets[1] = 0, offsets[2] = +1
                neighbor = tuple(cell[d] + offsets[offset_idx[d]] for d in range(len(cell)))
                yield neighbor
        
        for cell in active_cells_list:
            if cell not in visited:
                stack = [cell]
                cluster = []
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    cluster.append(cur)
                    for nb in neighbors(cur):
                        if nb in self.active_cells_ and nb not in visited:
                            stack.append(nb)
                clusters.append(cluster)
        
        # 给每个样本打标签
        labels = np.full(n_samples, -1, dtype=int)
        for cluster_id, cluster_cells in enumerate(clusters):
            cell_set = set(cluster_cells)
            in_cluster = [i for i, idx_tuple in enumerate(self.cell_idx_) 
                          if tuple(idx_tuple) in cell_set]
            labels[in_cluster] = cluster_id
        self.labels_ = labels
        
        # 计算每个聚类的质心
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        cluster_centers = []
        for cid in unique_clusters:
            points_in_cluster = X[labels == cid]
            center = points_in_cluster.mean(axis=0)
            cluster_centers.append(center)
        self.cluster_centers_ = np.array(cluster_centers) if len(cluster_centers) > 0 else np.empty((0, n_features))
        
        return self

    def predict(self, X):
        if self.cluster_centers_.shape[0] == 0:
            return np.full(X.shape[0], -1, dtype=int)
        from sklearn.metrics import pairwise_distances_argmin
        labels = pairwise_distances_argmin(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


##########################################
# --------- 评估指标与工具函数 -----------
##########################################

def separation(data, cluster_centers):
    if len(cluster_centers) <= 1:
        # 若只有一个簇或无簇，则将分离度返回为0
        return 0.0
    pairwise_dist = pairwise_distances(cluster_centers)
    np.fill_diagonal(pairwise_dist, np.inf)
    return pairwise_dist.min()

def compactness(data, labels, cluster_centers):
    if len(cluster_centers) == 0:
        return 0.0
    total_distance = 0
    count = 0
    for cluster_id in range(len(cluster_centers)):
        cluster_samples = data[labels == cluster_id]
        if len(cluster_samples) == 0:
            continue
        distances = np.linalg.norm(cluster_samples - cluster_centers[cluster_id], axis=1)
        total_distance += distances.sum()
        count += len(cluster_samples)
    if count == 0:
        return 0.0
    return total_distance / count

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

def evaluate_clustering(true_labels, pred_labels, data, cluster_centers):
    # LabelEncoder 保持和 kmeans 版本一致
    label_encoder = LabelEncoder()
    encoded_true_labels = label_encoder.fit_transform(true_labels)
    
    # 若 pred_labels 中全部为 -1，说明无有效聚类
    if np.all(pred_labels == -1):
        accuracy = 0.0
        ari = 0.0
        silhouette = -1.0
        comp = 0.0
        sep = 0.0
        return accuracy, ari, silhouette, comp, sep

    accuracy = cluster_accuracy(encoded_true_labels, pred_labels)
    ari = adjusted_rand_score(encoded_true_labels, pred_labels)
    
    # 当所有点都在同一个簇时，silhouette_score 会报错，这里做一下异常处理
    n_clusters_ = len(np.unique(pred_labels))
    if n_clusters_ < 2 or n_clusters_ == len(data):
        silhouette = -1
    else:
        silhouette = silhouette_score(data, pred_labels)
    
    comp = compactness(data, pred_labels, cluster_centers)
    sep = separation(data, cluster_centers)
    
    return accuracy, ari, silhouette, comp, sep


##########################################
# ------------ Optuna 优化 --------------
##########################################

def objective(trial, train_data, features):
    start_time = time.time()
    
    variance_threshold = trial.suggest_float('variance_threshold', 0, 0.2)
    n_components = trial.suggest_float('n_components', 0.1, 1.0)
    
    # STING 的超参数
    grid_size = trial.suggest_int('grid_size', 2, 8)
    density_threshold = trial.suggest_float('density_threshold', 0.0, 0.05)
    
    # 固定为 6 类主要是和原 HAR 数据集标签对应做评估
    # 但 STING 不一定会恰好分成 6 簇
    # 这里只是保留原 KMeans 里的 n_clusters=6 的注释
    n_clusters = 6
    
    # 数据预处理
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    X_train = pp.fit_transform(train_data[features])
    true_labels = train_data['activity']

    # 使用 STING 进行聚类
    sting = STING(grid_size=grid_size, density_threshold=density_threshold)
    pred_labels = sting.fit_predict(X_train)
    
    # 评估
    accuracy, ari, silhouette, comp, sep = evaluate_clustering(
        true_labels, pred_labels, X_train, sting.cluster_centers_
    )

    end_time = time.time()
    print(f"Trial finished in {end_time - start_time:.4f} seconds")
    
    # 用 accuracy 作为优化目标
    return accuracy


if __name__ == "__main__":
    # 读取数据
    train_data, test_data, features = read_data()
    
    study = optuna.create_study(direction='maximize')
    trial_times = []

    def callback(study, trial):
        trial_times.append(trial.duration.total_seconds())
        print(f"Trial {trial.number} runtime: {trial_times[-1]:.4f} seconds")

    study.optimize(partial(objective, train_data=train_data, features=features), 
                   n_trials=20, callbacks=[callback])

    best_params = study.best_params
    best_value = study.best_value
    print(f"Best Parameters: {best_params}")
    print(f"Best Value (Accuracy): {best_value:.4f}")

    # ------------------ 用最优参数再次训练并在训练、测试集上评估 ------------------
    variance_threshold = best_params['variance_threshold']
    n_components = best_params['n_components']
    grid_size = best_params['grid_size']
    density_threshold = best_params['density_threshold']
    
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    
    X_train = pp.fit_transform(train_data[features])
    true_train_labels = train_data['activity']
    X_test = pp.transform(test_data[features])
    true_test_labels = test_data['activity']

    sting = STING(grid_size=grid_size, density_threshold=density_threshold)
    pred_train_labels = sting.fit_predict(X_train)
    pred_test_labels = sting.predict(X_test)  # 注意 predict

    # 可视化训练集结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], 
                    hue=pred_train_labels, palette="Set1", s=60)
    plt.title("STING Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    os.makedirs("result/ModelResults", exist_ok=True)
    plt.savefig("result/ModelResults/sting_train.pdf", bbox_inches='tight')
    plt.close()

    # 可视化测试集结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], 
                    hue=pred_test_labels, palette="Set1", s=60)
    plt.title("STING Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("result/ModelResults/sting_test.pdf", bbox_inches='tight')
    plt.close()

    # 计算最终指标
    accuracy_train, ari_train, silhouette_train, comp_train, sep_train = evaluate_clustering(
        true_train_labels, pred_train_labels, X_train, sting.cluster_centers_
    )
    accuracy_test, ari_test, silhouette_test, comp_test, sep_test = evaluate_clustering(
        true_test_labels, pred_test_labels, X_test, sting.cluster_centers_
    )

    runtime_mean = np.mean(trial_times)
    runtime_variance = np.var(trial_times)

    print("Train Set Results:")
    print(f"  Accuracy: {round(accuracy_train, 4)}")
    print(f"  Adjusted Rand Score: {round(ari_train, 4)}")
    print(f"  Compactness: {round(comp_train, 4)}")
    print(f"  Separation: {round(sep_train, 4)}")
    print(f"  Silhouette Score: {round(silhouette_train, 4)}")

    print("Test Set Results:")
    print(f"  Accuracy: {round(accuracy_test, 4)}")
    print(f"  Adjusted Rand Score: {round(ari_test, 4)}")
    print(f"  Compactness: {round(comp_test, 4)}")
    print(f"  Separation: {round(sep_test, 4)}")
    print(f"  Silhouette Score: {round(silhouette_test, 4)}")

    print(f"Runtime Mean: {round(runtime_mean, 4)}")
    print(f"Runtime Variance: {round(runtime_variance, 4)}")