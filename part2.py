# Import definition
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Constants definition
GROUP_NUMBER = 16

# Resources
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# ------------------------------------------------------ FUNC -------------------------------------------------------- #


def flatten(list_of_lists) -> list:
    return list(val for sublist in list_of_lists for val in sublist)


def error_classification_rate(label, target, n_clusters) -> float:
    """
    * Calculates the error classification rate for a set of predictions made by a model.
    :param label: predicted values -> list
    :param target: real values -> list
    :param n_clusters: number of clusters used -> int
    :return: error calculation rate value -> float
    """
    def get_inter(t_val):
        """
        * Calculates interception between target and label's values
        :param t_val: target value to test -> (0 = benign or 1 = malignant)
        :return: number of occurrences -> int
        """
        return len([x for i, x in enumerate(label) if x == target[i] and target[i] == t_val])
    return sum([label.count(cluster) - max(get_inter(0), get_inter(1)) for cluster in range(n_clusters)]) / n_clusters



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


# Creates K-Means cluster with k = 2 and 3
kmeans_2 = KMeans(n_clusters=2, random_state=GROUP_NUMBER)
kmeans_3 = KMeans(n_clusters=3, random_state=GROUP_NUMBER)

# Trains both models and gets predicted labels for each one
label_2 = list(kmeans_2.fit_predict(X))
label_3 = list(kmeans_3.fit_predict(X))

# Calculates ECR value
print(f"ECR for K=2: {error_classification_rate(list(label_2), flatten(y), 2)}")
print(f"ECR for K=3: {error_classification_rate(list(label_3), flatten(y), 3)}")

# Calculates silhouette for each kmeans cluster
print(silhouette_score(X, label_2))
print(silhouette_score(X, label_3))


# --------------------------------------------------- QUESTION 5 ----------------------------------------------------- #
