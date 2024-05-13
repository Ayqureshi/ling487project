import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction

plt.close('all')

# Load the data
AllVCD = np.load("/Users/ameenqureshi/Desktop/CDs_D_E_T.npy")

# Number of clusters
num_clusters = 5  # Adjust based on your analysis needs

# Prepare data for clustering
data_for_clustering = []
for d in range(8):
    for token in range(20):
        for e in range(25):
            features = np.abs(AllVCD[e, :, d, token])
            if np.any(features > 0.5):  # Only consider encoder layers with significant features
                data_for_clustering.append(features)

data_for_clustering = np.array(data_for_clustering)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)  # Reduce to two dimensions for visualization
reduced_data = pca.fit_transform(data_for_clustering)

# Perform K-means clustering on reduced data
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(reduced_data)

# Visualize the clustering
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b', 'y', 'c']
for i in range(num_clusters):
    cluster_data = reduced_data[kmeans.labels_ == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i}', alpha=0.6)
plt.title('Cluster Visualization in 2D PCA-reduced space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
