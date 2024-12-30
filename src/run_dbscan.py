from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import optuna
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import read_data
from scipy.optimize import linear_sum_assignment

# 聚类评估函数
def evaluate_clustering(true_labels, pred_labels, data):
    # 如果所有点都被标记为噪声（-1），返回默认值
    if len(set(pred_labels) - {-1}) == 0:
        return 0, 0, -1  # 无法计算有效的ARI或Silhouette

    # Accuracy（基于重新分配的标签）
    label_encoder = LabelEncoder()
    encoded_true_labels = label_encoder.fit_transform(true_labels)
    accuracy = cluster_accuracy(encoded_true_labels, pred_labels)
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(encoded_true_labels, pred_labels)
    
    # Silhouette Score
    silhouette = silhouette_score(data, pred_labels) if len(set(pred_labels)) > 1 else -1  # 至少2个聚类才能计算
    
    return accuracy, ari, silhouette

# 聚类准确性计算（基于分配矩阵）
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

# Optuna目标函数
def objective(trial, train_data, features):
    # 超参数搜索
    variance_threshold = trial.suggest_float('variance_threshold', 0, 0.2)
    n_components = trial.suggest_float('n_components', 0.8, 1.0)
    eps = trial.suggest_float('eps', 10, 1000)
    min_samples = trial.suggest_int('min_samples', 4, 8)

    # 数据预处理流水线
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    X_train = pp.fit_transform(train_data[features])
    true_labels = train_data['activity']
    
    # 聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    pred_labels = dbscan.fit_predict(X_train)

    # 评估指标
    accuracy, ari, silhouette = evaluate_clustering(true_labels, pred_labels, X_train)
    
    return silhouette  # 优化目标

# 主程序
if __name__ == "__main__":
    # 读取数据
    train_data, test_data, features = read_data()
    
    # 优化
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, train_data=train_data, features=features), n_trials=100)

    # 获取最优参数
    best_params = study.best_params
    variance_threshold = best_params['variance_threshold']
    n_components = best_params['n_components']
    eps = best_params['eps']
    min_samples = best_params['min_samples']

    # 数据预处理
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    X_train = pp.fit_transform(train_data[features])
    true_train_labels = train_data['activity']

    X_test = pp.transform(test_data[features])
    true_test_labels = test_data['activity']

    # DBSCAN模型
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    pred_train_labels = dbscan.fit_predict(X_train)
    pred_test_labels = dbscan.fit_predict(X_test)

    # 可视化结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=pred_train_labels, palette="Set1", s=60)
    plt.title("DBSCAN Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/dbscan_train.pdf")

    # 测试集可视化
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=pred_test_labels, palette="Set1", s=60)
    plt.title("DBSCAN Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/dbscan_test.pdf")

    # 计算最终指标
    accuracy_train, ari_train, silhouette_train = evaluate_clustering(
        true_train_labels, pred_train_labels, X_train
    )
    accuracy_test, ari_test, silhouette_test = evaluate_clustering(
        true_test_labels, pred_test_labels, X_test
    )

    print(f"Best Parameters: {best_params}")
    print("Train Set Results:")
    print(f"  Accuracy: {round(accuracy_train, 4)}")
    print(f"  Adjusted Rand Score: {round(ari_train, 4)}")
    print(f"  Silhouette Score: {round(silhouette_train, 4)}")
    
    print("Test Set Results:")
    print(f"  Accuracy: {round(accuracy_test, 4)}")
    print(f"  Adjusted Rand Score: {round(ari_test, 4)}")
    print(f"  Silhouette Score: {round(silhouette_test, 4)}")