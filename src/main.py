import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score, adjusted_rand_score, confusion_matrix, pairwise_distances
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sympy import true
from clarans import CLARANS
from kmeans import ManualKMeans

np.random.seed(42)


def cluster_accuracy(true_labels, labels):
    conf_matrix = confusion_matrix(true_labels, labels)
    print(conf_matrix)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    print(row_ind, col_ind)
    mapping = dict(zip(col_ind, row_ind))
    mapped_labels = np.array([mapping[label] for label in labels])
    accuracy = accuracy_score(true_labels, mapped_labels)
    return accuracy

def compute_compactness_separation(X, labels):
    # 紧密度(簇内平方和)
    unique_labels = np.unique(labels)
    compactness = 0.0
    centers = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_center = cluster_points.mean(axis=0)  # 计算质心
        centers.append(cluster_center)
        compactness += np.sum((cluster_points - cluster_center) ** 2)  # 簇内距离总和
    
    n_clusters = len(unique_labels)
    sep = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            sep += np.linalg.norm(centers[i] - centers[j])
    separation = sep / (n_clusters * (n_clusters - 1) / 2)
    return compactness, separation


def compactness(X, labels, centers):
    return np.sum([np.sum((X[labels == i] - centers[i])**2) for i in range(len(centers))])

def plot_k_distance(X, k=5):
    # 计算每个点到其第 k 个最近邻的距离
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # 计算每个点的 k 距离，并进行排序
    k_distances = np.sort(distances[:, k-1], axis=0)
    
    # 绘制 k-distance 图
    plt.figure(figsize=(8, 6))
    plt.plot(k_distances)
    plt.title(f'k-distance Plot (k={k})')
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.grid(True)
    plt.savefig("result/knn.pdf")

def separation(centers):
    n_clusters = len(centers)
    sep = 0
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            sep += np.linalg.norm(centers[i] - centers[j])
    return sep / (n_clusters * (n_clusters - 1) / 2)

def read_data():
    dataset_path = './data'
    # 读取特征
    features = pd.read_csv(os.path.join(dataset_path, 'features.txt'), sep='\s+', header=None, names=['index', 'feature_name'])
    features = features['feature_name'].values
    for i in range(len(features)):
        features[i] = str(i)+"-"+features[i]

    # 读取活动标签
    activity_labels = pd.read_csv(os.path.join(dataset_path, 'activity_labels.txt'), sep='\s+', header=None, names=['index', 'activity'])

    # 读取训练数据
    X_train = pd.read_csv(os.path.join(dataset_path, 'train', 'X_train.txt'), delim_whitespace=True, header=None, names=features)
    Y_train = pd.read_csv(os.path.join(dataset_path, 'train', 'y_train.txt'), header=None, names=['activity'])
    subject_train = pd.read_csv(os.path.join(dataset_path, 'train', 'subject_train.txt'), header=None, names=['subject'])
    train_data = pd.concat([subject_train, Y_train, X_train], axis=1)
    print(train_data.shape)

    # 读取测试数据
    X_test = pd.read_csv(os.path.join(dataset_path, 'test', 'X_test.txt'), delim_whitespace=True, header=None, names=features)
    Y_test = pd.read_csv(os.path.join(dataset_path, 'test', 'y_test.txt'), header=None, names=['activity'])
    subject_test = pd.read_csv(os.path.join(dataset_path, 'test', 'subject_test.txt'), header=None, names=['subject'])
    test_data = pd.concat([subject_test, Y_test, X_test], axis=1)
    print(test_data.shape)


    # 合并训练和测试数据
    all_data = pd.concat([train_data, test_data], axis=0)
    all_data['activity'] = all_data['activity'].map(activity_labels.set_index('index')['activity'])

    print(f"Null counts in data: {all_data.isnull().sum().sum()}")

    return train_data, test_data, features

def preprocess(train_data, test_data, variance_threshold, n_components):
    pp = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=variance_threshold)),
        ('pca', PCA(n_components=n_components))
    ])
    X_train = pp.fit_transform(train_data)
    X_test = pp.fit_transform(test_data)
    return X_train, X_test

def dbscan_cluster(train_data_pca, test_data_pca, eps=5, min_samples=5):
    # 聚类部分（使用DBSCAN）
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    train_clusters = dbscan.fit_predict(train_data_pca)

    # 可视化聚类结果（训练集）
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], hue=train_clusters, palette="Set1", s=60)
    plt.title("DBSCAN Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/dbscan_fit_train.pdf")

    if len(set(train_clusters)) > 1:  # 如果有多个簇
        silhouette_avg = silhouette_score(train_data_pca, train_clusters)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
    else:
        print("Silhouette Score: Undefined for a single cluster")

    # 聚类紧凑度和分离度
    unique_clusters = set(train_clusters)
    if -1 in unique_clusters:  # 忽略噪声点
        unique_clusters.remove(-1)

    if len(unique_clusters) > 1:  # 如果簇的数量大于1，计算分离度
        centers = np.array([train_data_pca[train_clusters == i].mean(axis=0) for i in unique_clusters])
        compact = compactness(train_data_pca, train_clusters, centers)
        separate = separation(centers)

        print(f"Compactness: {compact:.3f}")
        print(f"Separation: {separate:.3f}")
    else:
        print("Separation: Undefined for a single cluster or noise-only clusters")

    # 对测试集进行预测
    test_clusters = dbscan.fit_predict(test_data_pca)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_data_pca[:, 0], y=test_data_pca[:, 1], hue=test_clusters, palette="Set1", s=60)
    plt.title("DBSCAN Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/dbscan_fit_test.pdf")

    # 评估聚类的分类准确率和调整后的兰德指数
    le = LabelEncoder()
    true_labels = le.fit_transform(test_data['activity'])

    # Calculate the size of each cluster
    cluster_sizes = {i: np.sum(test_clusters == i) for i in set(test_clusters)}
    # Sort clusters by size and get the top 6
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    top_6_clusters = dict(sorted_clusters[:6])
    print(f"cluster sizes: {cluster_sizes}")

    cluster_labels = np.zeros_like(test_clusters)
    for i in top_6_clusters.keys():
        # Get the most common label in the current cluster
        mask = (test_clusters == i)
        most_common_label = mode(true_labels[mask])[0]
        cluster_labels[test_clusters == i] = most_common_label

    # Accuracy is calculated only for the top 6 largest clusters, but the denominator remains the entire dataset
    accuracy = accuracy_score(true_labels, cluster_labels)
    print(f"Classification Accuracy for Top 6 Largest Clusters: {accuracy:.3f}")
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

def em_cluster(train_data_pca, test_data_pca, n_components=6):
    # 使用高斯混合模型（EM算法）
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    train_clusters = gmm.fit_predict(train_data_pca)

    # 可视化聚类结果（训练集）
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], size=3, hue=train_clusters, palette="Set1", s=60)
    plt.title("EM (Gaussian Mixture) Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/em_fit_train.pdf")

    # 计算聚类效果评估指标（如轮廓系数）
    silhouette_avg = silhouette_score(train_data_pca, train_clusters)
    print(f"Silhouette Score: {silhouette_avg:.3f}")

    # 聚类紧凑度和分离度
    centers = gmm.means_  # 获取各簇的中心
    compact = compactness(train_data_pca, train_clusters, centers)
    separate = separation(centers)

    print(f"Compactness: {compact:.3f}")
    print(f"Separation: {separate:.3f}")

    # 对测试集进行预测
    test_clusters = gmm.predict(test_data_pca)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_data_pca[:, 0], y=test_data_pca[:, 1], size=3, hue=test_clusters, palette="Set1", s=60)
    plt.title("EM (Gaussian Mixture) Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/em_fit_test.pdf")

    le = LabelEncoder()
    true_labels = le.fit_transform(test_data['activity'])

    # 计算每个簇的大小
    cluster_sizes = {i: np.sum(test_clusters == i) for i in set(test_clusters)}
    # 根据簇的大小排序并选取前6个最大簇
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    top_6_clusters = dict(sorted_clusters[:6])
    print(f"Cluster sizes: {cluster_sizes}")

    cluster_labels = np.zeros_like(test_clusters)
    for i in top_6_clusters.keys():
        # 获取当前簇内的最常见标签
        mask = (test_clusters == i)
        most_common_label = mode(true_labels[mask])[0]
        cluster_labels[test_clusters == i] = most_common_label

    # 计算分类准确率（只计算最大6个类的准确率，分母为全体样本）
    accuracy = accuracy_score(true_labels, cluster_labels)
    print(f"Classification Accuracy for Top 6 Largest Clusters: {accuracy:.3f}")
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

def kmeans_cluster(train_data_pca, test_data_pca):
 
    # 聚类部分（使用K-means）
    kmeans = ManualKMeans(n_clusters=6, max_iter=300)
    # kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, init='random')
    kmeans.fit(train_data_pca)
    train_clusters = kmeans.predict(train_data_pca)
    # train_clusters = kmeans.fit_predict(train_data_pca)

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], hue=train_clusters, size=3, palette="Set1", s=60)
    plt.title("K-means Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/kmeans_fit_train.pdf")

    # 计算聚类效果评估指标（如轮廓系数）
    silhouette_avg = silhouette_score(train_data_pca, train_clusters)
    print(f"Silhouette Score: {silhouette_avg:.3f}")

    # 聚类紧凑度和分离度
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(train_data_pca)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    compact = compactness(train_data_pca, labels, centers)
    separate = separation(centers)
    print(f"Compactness: {compact:.3f}")
    print(f"Separation: {separate:.3f}")

    test_clusters = kmeans.predict(test_data_pca)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_data_pca[:, 0], y=test_data_pca[:, 1], hue=test_clusters, size=3, palette="Set1", s=60)
    plt.title("K-means Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/ModelResults/kmeans_fit_test.pdf")


    le = LabelEncoder()
    true_labels = le.fit_transform(test_data['activity'])

    # cluster_labels = np.zeros_like(test_clusters)
    # for i in range(kmeans.n_clusters):
    #     # 获取该簇内样本的真实标签
    #     mask = (test_clusters == i)
    #     most_common_label = mode(true_labels[mask])[0]
    #     cluster_labels[test_clusters == i] = most_common_label

    # accuracy = accuracy_score(true_labels, cluster_labels)
    accuracy = cluster_accuracy(true_labels, test_clusters)
    print(f"Classification Accuracy: {accuracy:.3f}")
    ari = adjusted_rand_score(true_labels, test_clusters)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

def clarans_cluster(train_data_pca, test_data_pca, max_neighbors=5, max_swap=5, n_clusters=6):
    # CLARANS聚类
    clarans = CLARANS(n_clusters=n_clusters, max_neighbors=max_neighbors, max_swap=max_swap, random_state=42)
    train_clusters = clarans.fit_predict(train_data_pca)

    # 可视化聚类结果（训练集）
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], hue=train_clusters, size=3, palette="Set1", s=60)
    plt.title("CLARANS Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/clarans_fit_train.pdf")

    # 计算聚类效果评估指标（如轮廓系数）
    silhouette_avg = silhouette_score(train_data_pca, train_clusters)
    print(f"Silhouette Score: {silhouette_avg:.3f}")

    # 聚类紧凑度和分离度
    centers = clarans.cluster_centers_  # 获取各簇的中心
    compact = compactness(train_data_pca, train_clusters, centers)
    separate = separation(centers)

    print(f"Compactness: {compact:.3f}")
    print(f"Separation: {separate:.3f}")

    # 对测试集进行预测
    test_clusters = clarans.predict(test_data_pca)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_data_pca[:, 0], y=test_data_pca[:, 1], hue=test_clusters, size=3, palette="Set1", s=60)
    plt.title("CLARANS Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/clarans_fit_test.pdf")

    le = LabelEncoder()
    true_labels = le.fit_transform(test_data['activity'])

    # 计算每个簇的大小
    cluster_sizes = {i: np.sum(test_clusters == i) for i in set(test_clusters)}
    # 根据簇的大小排序并选取前6个最大簇
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    top_6_clusters = dict(sorted_clusters[:6])
    print(f"Cluster sizes: {cluster_sizes}")

    cluster_labels = np.zeros_like(test_clusters)
    for i in top_6_clusters.keys():
        # 获取当前簇内的最常见标签
        mask = (test_clusters == i)
        most_common_label = mode(true_labels[mask])[0]
        cluster_labels[test_clusters == i] = most_common_label

    # 计算分类准确率（只计算最大6个类的准确率，分母为全体样本）
    accuracy = accuracy_score(true_labels, cluster_labels)
    print(f"Classification Accuracy for Top 6 Largest Clusters: {accuracy:.3f}")
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

def denclue_cluster(train_data_pca, test_data_pca, bandwidth=10):
    """
    使用核密度估计 (Kernel Density Estimation, KDE) 作为 DENCLUE 聚类的一种方式
    """
    # 训练集的密度估计
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(train_data_pca)

    # 对训练集数据进行评估，返回每个点的密度估计值
    train_density = kde.score_samples(train_data_pca)
    # 通过密度值进行排序
    train_clusters = np.digitize(train_density, np.percentile(train_density, [16.67, 33.33, 50, 66.67, 83.33, 100]))

    # 可视化训练集的聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], hue=train_clusters, palette="Set1", s=60)
    plt.title("DENCLUE (Density Clustering) of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/denclue_fit_train.pdf")

    # 对测试集进行密度估计和聚类
    test_density = kde.score_samples(test_data_pca)
    test_clusters = np.digitize(test_density, np.percentile(train_density, [16.67, 33.33, 50, 66.67, 83.33, 100]))

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_data_pca[:, 0], y=test_data_pca[:, 1], hue=test_clusters, palette="Set1", s=60)
    plt.title("DENCLUE (Density Clustering) of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/denclue_fit_test.pdf")

    # 计算分类准确率，仅计算最大 6 个类的准确率，分母为全体样本
    le = LabelEncoder()
    true_labels = le.fit_transform(test_data['activity'])

    # 计算每个簇的大小
    cluster_sizes = {i: np.sum(test_clusters == i) for i in set(test_clusters)}
    # 排序并选取前 6 个最大簇
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    top_6_clusters = dict(sorted_clusters[:6])
    print(f"Cluster sizes: {cluster_sizes}")

    # 为每个簇分配最常见的标签
    cluster_labels = np.zeros_like(test_clusters)
    for i in top_6_clusters.keys():
        mask = (test_clusters == i)
        most_common_label = mode(true_labels[mask])[0]
        cluster_labels[test_clusters == i] = most_common_label

    # 计算准确率
    accuracy = accuracy_score(true_labels, cluster_labels)
    print(f"Classification Accuracy for Top 6 Largest Clusters: {accuracy:.3f}")
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

def agglomerative_cluster(train_data_pca, test_data_pca):
 
    # 聚类部分
    agg = AgglomerativeClustering(n_clusters=6, linkage='ward')
    pred_labels_train = agg.fit_predict(train_data_pca)

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], hue=pred_labels_train, size=3, palette="Set1", s=60)
    plt.title("Agglomerative Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/agg_fit_train.pdf")

    # 计算聚类效果评估指标（如轮廓系数）
    silhouette_avg = silhouette_score(train_data_pca, pred_labels_train)
    print(f"Silhouette Score: {silhouette_avg:.3f}")

    # 聚类紧凑度和分离度
    compact, separate = compute_compactness_separation(train_data_pca, pred_labels_train)
    print(f"Compactness: {compact:.3f}")
    print(f"Separation: {separate:.3f}")

    pred_labels_test = agg.fit_predict(test_data_pca)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_data_pca[:, 0], y=test_data_pca[:, 1], hue=pred_labels_test, size=3, palette="Set1", s=60)
    plt.title("Agglomerative Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/agg_fit_test.pdf")


    le = LabelEncoder()
    true_labels_test = le.fit_transform(test_data['activity'])
    accuracy = cluster_accuracy(true_labels_test, pred_labels_test)
    print(f"Classification Accuracy: {accuracy:.3f}")
    ari = adjusted_rand_score(true_labels_test, pred_labels_test)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

def divisive_cluster(train_data_pca, test_data_pca):
    def divisive_clustering(X, n_clusters):
        # 初始时所有点属于同一个簇
        clusters = [X]
        labels = np.zeros(len(X), dtype=int)
        current_label = 1

        while len(clusters) < n_clusters:
            # 找到最大簇进行分裂
            largest_cluster_idx = np.argmax([len(cluster) for cluster in clusters])
            largest_cluster = clusters.pop(largest_cluster_idx)

            # 使用 K-Means 将该簇分裂为 2 个子簇
            kmeans = KMeans(n_clusters=2, random_state=42).fit(largest_cluster)
            sub_labels = kmeans.labels_

            # 更新簇和标签
            for i in range(2):
                clusters.append(largest_cluster[sub_labels == i])
                labels[np.isin(X, largest_cluster[sub_labels == i]).all(axis=1)] = current_label
                current_label += 1

        return labels
    pred_labels_train = divisive_clustering(train_data_pca, n_clusters=6)
    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=train_data_pca[:, 0], y=train_data_pca[:, 1], hue=pred_labels_train, size=3, palette="Set1", s=60)
    plt.title("Divisive Clustering of UCI HAR Dataset (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/div_fit_train.pdf")

    # 计算聚类效果评估指标（如轮廓系数）
    silhouette_avg = silhouette_score(train_data_pca, pred_labels_train)
    print(f"Silhouette Score: {silhouette_avg:.3f}")

    # 聚类紧凑度和分离度
    compact, separate = compute_compactness_separation(train_data_pca, pred_labels_train)
    print(f"Compactness: {compact:.3f}")
    print(f"Separation: {separate:.3f}")

    pred_labels_test = divisive_clustering(test_data_pca, 6)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=test_data_pca[:, 0], y=test_data_pca[:, 1], hue=pred_labels_test, size=3, palette="Set1", s=60)
    plt.title("Divisive Clustering of UCI HAR Dataset (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.savefig("result/div_fit_test.pdf")

    le = LabelEncoder()
    true_labels_test = le.fit_transform(test_data['activity'])
    pred_labels_test = le.fit_transform(pred_labels_test)
    accuracy = cluster_accuracy(true_labels_test, pred_labels_test)
    print(f"Classification Accuracy: {accuracy:.3f}")
    ari = adjusted_rand_score(true_labels_test, pred_labels_test)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")






if __name__ == "__main__":
    train_data, test_data, features = read_data()
    train_data_pca, test_data_pca = preprocess(train_data[features], test_data[features], 0.1, 0.8)
    # plot_k_distance(train_data_pca, k=5)
    # kmeans_cluster(train_data_pca, test_data_pca)
    dbscan_cluster(train_data_pca, test_data_pca)
    # em_cluster(train_data_pca, test_data_pca)
    # clarans_cluster(train_data_pca, test_data_pca)
    # denclue_cluster(train_data_pca, test_data_pca)
    # agglomerative_cluster(train_data_pca, test_data_pca)
    # divisive_cluster(train_data_pca, test_data_pca)