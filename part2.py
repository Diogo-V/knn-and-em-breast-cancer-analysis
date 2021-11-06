# Import definition
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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

i = 1

for n_clusters in range_n_clusters:

    # Creates K-Means cluster with k = n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=GROUP_NUMBER)

    # Trains model and gets predicted labels
    cluster_labels = kmeans.fit_predict(X)

    # Prints ECR result for this cluster
    print(f"ECR for K={n_clusters}: {error_classification_rate(list(cluster_labels), flatten(y), n_clusters)}")

    # Calculates silhouette for each kmeans cluster
    print(f"Silhouette for K={n_clusters}: {silhouette_score(X, cluster_labels)}")

    plt.figure(figsize=(12, 12))

    index_1 = 2*10**2 + 2*10 + i

    plt.subplot(index_1)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=16)
    plt.title('The visualization of the clustered data')

    index_2 = 2*10**2 + 2*10 + i + 1
    plt.subplot(index_2)
    visualizer = SilhouetteVisualizer(kmeans, colors='viridis')
    visualizer.fit(X)
    plt.xlabel('Cluster label', fontsize=18)
    plt.ylabel('Silhouete coefficient values', fontsize=16)
    plt.title("Silhouette plot for the various clusters")


plt.show()


# --------------------------------------------------- QUESTION 5 ----------------------------------------------------- #
