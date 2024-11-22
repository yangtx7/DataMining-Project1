import numpy as np

class ManualKMeans:
    def __init__(self, n_clusters=6, max_iter=300, tol=1e-4, random_state=None):
        """
        手动实现的 KMeans 聚类类。
        Args:
            n_clusters: 聚类数
            max_iter: 最大迭代次数
            tol: 收敛阈值
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # 随机初始化聚类中心
        initial_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[initial_indices]

        for iteration in range(self.max_iter):
            # 计算每个点到所有中心的距离
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

            # 分配点到最近的中心
            labels = np.argmin(distances, axis=1)

            # 计算新的中心
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # 检查是否收敛
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
