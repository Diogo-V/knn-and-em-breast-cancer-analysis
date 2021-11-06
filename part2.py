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
label_2 = kmeans_2.fit_predict(X)
label_3 = kmeans_3.fit_predict(X)

# Calculates silhouette for each kmeans cluster
silhouette_2 = silhouette_score(X, label_2)
silhouette_3 = silhouette_score(X, label_3)

print(silhouette_2)
print(silhouette_3)

# --------------------------------------------------- QUESTION 5 ----------------------------------------------------- #
