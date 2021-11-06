# Import definition
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
import numpy as np

# Constants definition
from yellowbrick.cluster import SilhouetteVisualizer

GROUP_NUMBER = 16

# Resources
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# ---------------------------------------- PREPROCESSING AND DATA FILTERING ------------------------------------------ #


# Loading dataset into working desk
data = arff.loadarff('./data/breast.w.arff')
df = pd.DataFrame(data[0])

# Removes NaN values from dataset by deleting rows
df.dropna(axis=0, how="any", inplace=True)

# Gets X (data matrix) and y (target values column matrix)
X = df.drop("Class", axis=1).to_numpy()
y = df["Class"].to_numpy()

# Perform some preprocessing by turning labels into binaries (benign is 1)
# We are doing a "double conversion" to convert everything to Binary type
for count, value in enumerate(y):
    if value == b"malignant":
        y[count] = "yes"
    else:
        y[count] = "no"

lb = LabelBinarizer()
y = lb.fit_transform(y)


# --------------------------------------------------- QUESTION 4 ----------------------------------------------------- #
range_n_clusters = [2, 3]

i = 1

for n_clusters in range_n_clusters:

    # Creates K-Means cluster with k = n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=GROUP_NUMBER)

    # Trains model and gets predicted labels
    cluster_labels = kmeans.fit_predict(X)

    # Calculates silhouette for each kmeans cluster
    silhouette = silhouette_score(X, cluster_labels)
    print("Silhouete for = ", n_clusters, "cluster is :", silhouette,)

    plt.figure(figsize=(12, 12))

    index_1 = 2*10**2 + 2*10 + i

    plt.subplot(index_1)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)
    plt.xlabel('Feature space for the 1st feature')
    plt.ylabel('Feature space for the 2nd feature')
    plt.title('The visualization of the clustered data')

    index_2 = 2*10**2 + 2*10 + i + 1
    plt.subplot(index_2)
    visualizer = SilhouetteVisualizer(kmeans, colors='viridis')
    visualizer.fit(X)
    plt.xlabel('Cluster label')
    plt.ylabel('Silhouete coefficient values')
    plt.title("Silhouette plot for the various clusters")


plt.show()


# --------------------------------------------------- QUESTION 5 ----------------------------------------------------- #
