import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, pairwise_distances
from sklearn.pipeline import Pipeline
import time
import optuna
from functools import partial
from main import read_data
from scipy.optimize import linear_sum_assignment

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

class KMeansManual:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, init='k-means++', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _init_centroids(self, data):
        np.random.seed(self.random_state)
        if self.init == 'random':
            indices = np.random.choice(len(data), self.n_clusters, replace=False)
            return data[indices]
        elif self.init == 'k-means++':
            centroids = [data[np.random.choice(len(data))]]
            for _ in range(1, self.n_clusters):
                distances = np.min(pairwise_distances(data, centroids), axis=1)
                probabilities = distances / distances.sum()
                next_centroid = data[np.random.choice(len(data), p=probabilities)]
                centroids.append(next_centroid)
            return np.array(centroids)
        else:
            raise ValueError(f"Unknown init method: {self.init}")

    def fit(self, data):
        self.cluster_centers_ = self._init_centroids(data)
        for i in range(self.max_iter):
            distances = pairwise_distances(data, self.cluster_centers_)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(self.n_clusters)])
            if np.linalg.norm(new_centroids - self.cluster_centers_) < self.tol:
                break
            self.cluster_centers_ = new_centroids
        self.labels_ = labels
        return self

    def predict(self, data):
        distances = pairwise_distances(data, self.cluster_centers_)
        return np.argmin(distances, axis=1)

# 其他函数和主程序保持一致，修改了KMeans调用部分为KMeansManual
def evaluate_clustering(true_labels, pred_labels, data, cluster_centers):
    label_encoder = LabelEncoder()
    encoded_true_labels = label_encoder.fit_transform(true_labels)
    accuracy = cluster_accuracy(encoded_true_labels, pred_labels)
    ari = adjusted_rand_score(encoded_true_labels, pred_labels)
    silhouette = silhouette_score(data, pred_labels)
    comp = compactness(data, pred_labels, cluster_centers)
    sep = separation(data, cluster_centers)
    return accuracy, ari, silhouette, comp, sep

if __name__ == "__main__":
    result = []

    train_data, test_data, features = read_data()
    
    study = optuna.create_study(direction='maximize')
    trial_times = []

    def callback(study, trial):
        trial_times.append(trial.duration.total_seconds())
        print(f"Trial {trial.number} runtime: {trial_times[-1]:.4f} seconds")

    def objective(trial, train_data, features):
        start_time = time.time()
        variance_threshold = trial.suggest_float('variance_threshold', 0, 0.2)
        n_components = trial.suggest_float('n_components', 0.7, 1.0)
        init = trial.suggest_categorical('init', ['k-means++', 'random'])
        n_clusters = 6

        pp = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', VarianceThreshold(threshold=variance_threshold)),
            ('pca', PCA(n_components=n_components)),
        ])
        X_train = pp.fit_transform(train_data[features])
        true_labels = train_data['activity']
        
        kmeans = KMeansManual(n_clusters=n_clusters, init=init, random_state=42)
        kmeans.fit(X_train)
        pred_labels = kmeans.labels_

        accuracy, ari, silhouette, comp, sep = evaluate_clustering(
            true_labels, pred_labels, X_train, kmeans.cluster_centers_
        )
        end_time = time.time()
        print(f"Trial finished in {end_time - start_time:.4f} seconds")
        return accuracy

    study.optimize(partial(objective, train_data=train_data, features=features), n_trials=20, callbacks=[callback])
    best_params = study.best_params

    variance_threshold = best_params['variance_threshold']
    n_components = best_params['n_components']
    init = best_params['init']
    n_clusters = 6

    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components)),
    ])
    X_train = pp.fit_transform(train_data[features])
    X_test = pp.transform(test_data[features])
    true_train_labels = train_data['activity']
    true_test_labels = test_data['activity']

    kmeans = KMeansManual(n_clusters=n_clusters, init=init, random_state=42)
    kmeans.fit(X_train)
    pred_train_labels = kmeans.labels_
    pred_test_labels = kmeans.predict(X_test)

    # 保存结果，修改前缀为kmeans-manual
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=pred_train_labels, palette="Set1", s=60)
    plt.title("KMeansManual Clustering (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/kmeans-manual_train.pdf")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=pred_test_labels, palette="Set1", s=60)
    plt.title("KMeansManual Clustering (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/kmeans-manual_test.pdf")

        # 计算最终指标
    accuracy_train, ari_train, silhouette_train, comp_train, sep_train = evaluate_clustering(
        true_train_labels, pred_train_labels, X_train, kmeans.cluster_centers_
    )
    accuracy_test, ari_test, silhouette_test, comp_test, sep_test = evaluate_clustering(
        true_test_labels, pred_test_labels, X_test, kmeans.cluster_centers_
    )

    runtime_mean = np.mean(trial_times)
    runtime_variance = np.var(trial_times)

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