import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class AmountClustersError(ValueError):
    """Кастомное исключение для некорректного числа кластеров."""

    def __init__(self, n_clusters: int, message="Число кластеров должно быть больше 0."):
        self.n_clusters = n_clusters
        self.message = message
        super().__init__(f"{message} Получено: {n_clusters}")


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4) -> None:
        """
        Инициализация модели K-Means (scikit-learn API style).

        Параметры:
            n_clusters (int): Количество кластеров.
            max_iter (int): Максимальное количество итераций алгоритма.
            tol (float): Порог сходимости (минимальное изменение центроид).
        """
        if n_clusters <= 0:
            raise AmountClustersError(n_clusters)

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, x: np.ndarray) -> 'KMeans':
        """Обучение модели с использованием полностью векторизованных операций NumPy."""
        x = np.asarray(x)

        random_indices = np.random.choice(x.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = x[random_indices].copy()

        labels = np.zeros(x.shape[0], dtype=int)

        for _ in range(self.max_iter):
            distances = np.linalg.norm(x[:, np.newaxis] - self.cluster_centers_, axis=2)

            labels = np.argmin(distances, axis=1)

            new_centers = np.zeros_like(self.cluster_centers_)

            for k in range(self.n_clusters):
                cluster_points = x[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = cluster_points.mean(axis=0)
                else:
                    new_centers[k] = x[np.random.randint(0, x.shape[0])]

            shift = np.sum(np.linalg.norm(new_centers - self.cluster_centers_, axis=1))
            self.cluster_centers_ = new_centers

            if shift < self.tol:
                break

        self.labels_ = labels
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Предсказание кластеров для новых данных."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод fit().")

        x = np.asarray(x)
        distances = np.linalg.norm(x[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

    @staticmethod
    def plot_clusters(
            x: np.ndarray,
            labels: np.ndarray,
            centers: np.ndarray,
            title: str = "K-Means Clustering"
    ):
        """Визуализация кластеров и их центроид."""
        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            x=x[:, 0], y=x[:, 1], hue=labels,
            palette="deep", s=50, alpha=0.8, legend='full'
        )

        plt.scatter(
            centers[:, 0], centers[:, 1],
            color='black', marker='X', s=200,
            label='Центроиды', zorder=10
        )

        plt.title(title, fontsize=14)
        plt.xlabel('Признак 1', fontsize=12)
        plt.ylabel('Признак 2', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


if __name__ == '__main__':
    n_clusters_true = 4

    X_data, _ = make_blobs(
        n_samples=1500, centers=n_clusters_true,
        cluster_std=0.8, random_state=42
    )

    kmeans = KMeans(n_clusters=n_clusters_true)
    kmeans.fit(X_data)

    predictions = kmeans.predict(X_data)

    KMeans.plot_clusters(
        X_data, predictions, kmeans.cluster_centers_,
        "Результаты векторизованного K-Means"
    )
