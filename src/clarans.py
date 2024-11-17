import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist

class CLARANS:
    def __init__(self, n_clusters=6, max_neighbors=5, max_swap=5, random_state=None):
        self.n_clusters = n_clusters
        self.max_neighbors = max_neighbors  # 最大邻居数（局部搜索的步数）
        self.max_swap = max_swap  # 最大交换次数
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        # 随机初始化簇中心
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        initial_centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        # 进行多次随机化搜索
        best_centers = initial_centers
        best_cost = np.inf
        for _ in range(self.max_neighbors):
            # 随机交换簇中心
            centers = self._perturb_centers(best_centers, X)
            # 计算当前簇中心的聚类代价
            cost = self._calculate_cost(X, centers)

            # 如果当前代价更低，则更新最优簇中心
            if cost < best_cost:
                best_cost = cost
                best_centers = centers

        # 返回最优的簇中心
        self.cluster_centers_ = best_centers
        self.labels_ = self._assign_labels(X, self.cluster_centers_)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        return self._assign_labels(X, self.cluster_centers_)

    def _perturb_centers(self, centers, X):
        """ 随机扰动簇中心 """
        new_centers = centers.copy()
        for _ in range(self.max_swap):
            # 随机选择一个簇中心并交换
            rand_cluster = np.random.randint(0, self.n_clusters)
            rand_sample = X[np.random.choice(X.shape[0])]
            new_centers[rand_cluster] = rand_sample
        return new_centers

    def _assign_labels(self, X, centers):
        """ 根据簇中心给每个样本分配标签 """
        # 计算每个样本到簇中心的距离
        distances = cdist(X, centers, 'euclidean')
        # 选取距离最近的簇中心
        labels = np.argmin(distances, axis=1)
        return labels

    def _calculate_cost(self, X, centers):
        """ 计算聚类的代价（所有样本到其簇中心的距离之和） """
        labels = self._assign_labels(X, centers)
        cost = 0
        for i in range(self.n_clusters):
            # 计算簇内所有样本到簇中心的距离
            cluster_points = X[labels == i]
            cost += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1))
        return cost
