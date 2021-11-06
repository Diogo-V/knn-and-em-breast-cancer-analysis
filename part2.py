# Import definition
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# Constants definition
from yellowbrick.cluster import SilhouetteVisualizer

GROUP_NUMBER = 16

# Resources
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# ------------------------------------------------------ FUNC -------------------------------------------------------- #


def flatten(list_of_lists) -> list:
    return list(val for sublist in list_of_lists for val in sublist)


def error_classification_rate(label, target, n_cls) -> float:
    """
    * Calculates the error classification rate for a set of predictions made by a model.
    :param label: predicted values -> list
    :param target: real values -> list
    :param n_cls: number of clusters used -> int
    :return: error calculation rate value -> float
    """
    def get_inter(t_val):
        """
        * Calculates interception between target and label's values
        :param t_val: target value to test -> (0 = benign or 1 = malignant)
        :return: number of occurrences -> int
        """
        return len([x for j, x in enumerate(label) if x == target[j] and target[j] == t_val])
    return sum([label.count(cluster) - max(get_inter(0), get_inter(1)) for cluster in range(n_cls)]) / n_cls


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

for n_clusters in range_n_clusters:

    # Creates K-Means cluster with k = n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=GROUP_NUMBER)

    # Trains model and gets predicted labels
    cluster_labels = kmeans.fit_predict(X)

    # Prints ECR result for this cluster
    print(f"ECR for K={n_clusters}: {error_classification_rate(list(cluster_labels), flatten(y), n_clusters)}")

    # Calculates silhouette for each kmeans cluster
    print(f"Silhouette for K={n_clusters}: {silhouette_score(X, cluster_labels)}")

    # Create figure canvas
    plt.figure(figsize=(12, 12))

    # Display silhouette coefficient for each sample in a cluster, then compares it to other clusters
    visualizer = SilhouetteVisualizer(kmeans, colors='viridis')

    # Fits the model
    visualizer.fit(X)

    # Plot tittle
    plt.xlabel('Cluster label')
    plt.ylabel('Silhouete coefficient values')
    plt.title("Silhouette plot for the various clusters")


plt.show()


# --------------------------------------------------- QUESTION 5 ----------------------------------------------------- #


# Selects the best k features using mutual information
X_new = SelectKBest(mutual_info_classif, k=2).fit_transform(X, y.ravel())

# Creates K-Means cluster with k = 3
kmeans_3 = KMeans(n_clusters=3, random_state=GROUP_NUMBER)

# Trains model and gets predicted labels
cluster_labels_3 = kmeans_3.fit_predict(X_new)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels_3, s=50,  cmap='viridis')

# Plot cluster centroids
centers = kmeans_3.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)

# Plot tittles
plt.xlabel('Feature space for the 1st feature')
plt.ylabel('Feature space for the 2nd feature')
plt.title('The visualization of the clustered data')

plt.show()
