import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class CustomKMeans:
    def __init__(self, k, max_iter=300, tol=1e-4):
        """
        Initialize the CustomKMeans class.
        :param k: Number of clusters
        :param max_iter: Maximum number of iterations
        :param tol: Convergence tolerance
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.cluster_assignments = None

    def fit(self, df):
        """
        Fit the k-means clustering algorithm to the dataframe.
        :param df: DataFrame to cluster
        """
        data = df.to_numpy()
        np.random.seed(42)
        self.centroids = data[np.random.choice(len(data), self.k, replace=False)]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            new_assignments = np.argmin(distances, axis=1)
            new_centroids = np.array([data[new_assignments == i].mean(axis=0) for i in range(self.k)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids
            self.cluster_assignments = new_assignments

    def predict(self, df):
        """
        Predict the cluster membership for each row in the dataframe.
        :param df: DataFrame to predict cluster IDs
        :return: List of cluster IDs
        """
        data = df.to_numpy()
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def sse(self):
        """
        Calculate the sum of squared errors (SSE) for the clusters.
        :return: SSE value
        """
        data = df_scaled.to_numpy()  # Operates on pre-fitted data
        distances = np.linalg.norm(data - self.centroids[self.cluster_assignments], axis=1)
        return np.sum(distances**2)


# Load the dataset
df = pd.read_csv("data/raw/demographic.csv")

# Select and standardize numeric columns for clustering
columns_to_cluster = ['WTINT2YR', 'WTMEC2YR', 'RIDAGEYR']
df_subset = df[columns_to_cluster]
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_subset), columns=columns_to_cluster)

# Run CustomKMeans for different k values
print("CustomKMeans Results:")
for k in range(2, 6):
    custom_kmeans = CustomKMeans(k)
    custom_kmeans.fit(df_scaled)
    print(f"CustomKMeans: k={k}, SSE={custom_kmeans.sse()}")

# Run and compare with sklearn's KMeans
print("\nSklearn KMeans Results:")
for k in range(2, 6):
    sklearn_kmeans = KMeans(n_clusters=k, random_state=42)
    sklearn_kmeans.fit(df_scaled)
    print(f"Sklearn KMeans: k={k}, SSE={sklearn_kmeans.inertia_}, Silhouette Score={silhouette_score(df_scaled, sklearn_kmeans.labels_)}")

# As k increases, the SSE decreases, however, the rate of decrease starts to slow significantly after k=3. 
# This suggests diminishing returns for increasing k beyond 3.
# The best k value is k=2 because it has the highest silhouette score and the closer a silhouette score is to one
# indicates good clustering.
