import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
import optuna
from functools import partial
import numpy as np
from main import read_data
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

def separation(data, cluster_centers):
    # 计算质心两两之间的距离
    pairwise_dist = pairwise_distances(cluster_centers)
    np.fill_diagonal(pairwise_dist, np.inf)  # 忽略自身到自身的距离（置为无穷大）
    # 返回最小距离作为分离度
    separation_value = pairwise_dist.min()
    return separation_value

def compactness(data, labels, cluster_centers):
    # 初始化总距离
    total_distance = 0
    # 计算每个簇内样本点与质心的距离总和
    for cluster_id in range(len(cluster_centers)):
        cluster_samples = data[labels == cluster_id]
        distances = np.linalg.norm(cluster_samples - cluster_centers[cluster_id], axis=1)
        total_distance += distances.sum()
    # 计算紧凑度
    compactness_value = total_distance / len(data)
    return compactness_value

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

# 定义计算聚类结果的指标函数
def evaluate_clustering(true_labels, pred_labels, data, cluster_centers):
    # Accuracy
    label_encoder = LabelEncoder()
    encoded_true_labels = label_encoder.fit_transform(true_labels)
    accuracy = cluster_accuracy(encoded_true_labels, pred_labels)
    
    # Adjusted Rand Score
    ari = adjusted_rand_score(encoded_true_labels, pred_labels)
    
    # Silhouette Score
    silhouette = silhouette_score(data, pred_labels)
    
    # Compactness and Separation
    comp = compactness(data, pred_labels, cluster_centers)
    sep = separation(data, cluster_centers)
    
    return accuracy, ari, silhouette, comp, sep

# 目标函数
def objective(trial, train_data, features):
    start_time = time.time()  # 开始计时

    # 超参数搜索
    variance_threshold = trial.suggest_float('variance_threshold', 0, 0.2)
    n_components = trial.suggest_float('n_components', 0.7, 1.0)
    init = trial.suggest_categorical('init', ['k-means++', 'random'])
    algo = trial.suggest_categorical('algorithm', ['lloyd', 'elkan', 'full'])
    n_clusters = 6  # 固定为 6 类

    # 数据预处理流水线
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    X_train = pp.fit_transform(train_data[features])
    true_labels = train_data['activity']
    
    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=42, n_init=10, algorithm=algo)
    pred_labels = kmeans.fit_predict(X_train)

    # 评估指标
    accuracy, ari, silhouette, comp, sep = evaluate_clustering(
        true_labels, pred_labels, X_train, kmeans.cluster_centers_
    )
    
    end_time = time.time()  # 结束计时
    trial_time = end_time - start_time
    print(f"Trial finished in {trial_time:.4f} seconds")

    # 返回 accuracy 作为优化目标
    return accuracy

import matplotlib.pyplot as plt
import seaborn as sns

# 主程序
if __name__ == "__main__":
    result = []

    # 读取数据
    train_data, test_data, features = read_data()
    
    # 优化
    study = optuna.create_study(direction='maximize')
    trial_times = []

    def callback(study, trial):
        """Callback function to record runtime for each trial."""
        trial_times.append(trial.duration.total_seconds())
        print(f"Trial {trial.number} runtime: {trial_times[-1]:.4f} seconds")

    study.optimize(partial(objective, train_data=train_data, features=features), n_trials=40, callbacks=[callback])

    # 获取最优参数
    best_params = study.best_params
    best_value = study.best_value

    # 用最优参数再次训练
    variance_threshold = best_params['variance_threshold']
    n_components = best_params['n_components']
    init = best_params['init']
    algo = best_params['algorithm']
    n_clusters = 6

    # 数据预处理
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    
    # 在训练集上拟合
    X_train = pp.fit_transform(train_data[features])
    true_train_labels = train_data['activity']

    # 在测试集上转换
    X_test = pp.transform(test_data[features])
    true_test_labels = test_data['activity']

    # 训练 KMeans 模型
    kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=42, n_init=10, algorithm=algo)
    kmeans.fit(X_train)
    
    # 在训练集和测试集上进行预测
    pred_train_labels = kmeans.predict(X_train)
    pred_test_labels = kmeans.predict(X_test)

    # 可视化训练集结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=pred_train_labels, palette="Set1", s=60)
    plt.title("KMeans Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/kmeans_train.pdf")

    # 可视化测试集结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=pred_test_labels, palette="Set1", s=60)
    plt.title("KMeans Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/kmeans_test.pdf")

    # 计算最终指标
    accuracy_train, ari_train, silhouette_train, comp_train, sep_train = evaluate_clustering(
        true_train_labels, pred_train_labels, X_train, kmeans.cluster_centers_
    )
    accuracy_test, ari_test, silhouette_test, comp_test, sep_test = evaluate_clustering(
        true_test_labels, pred_test_labels, X_test, kmeans.cluster_centers_
    )

    runtime_mean = np.mean(trial_times)
    runtime_variance = np.var(trial_times)

    # 输出结果
    print(f"Best Parameters: {best_params}")
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