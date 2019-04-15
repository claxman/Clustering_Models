# Author: Chaitanay Laxman

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_pred = ac.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_pred == 0,0], X[y_pred == 0,1], s = 40, c = "green", label = "cluster1")
plt.scatter(X[y_pred == 1,0], X[y_pred == 1,1], s = 40, c = "blue", label = "cluster2")
plt.scatter(X[y_pred == 2,0], X[y_pred == 2,1], s = 40, c = "grey", label = "cluster3")
plt.scatter(X[y_pred == 3,0], X[y_pred == 3,1], s = 40, c = "purple", label = "cluster4")
plt.scatter(X[y_pred == 4,0], X[y_pred == 4,1], s = 40, c = "black", label = "cluster5")

plt.title("Cluster of customers")
plt.xlabel("Annual Income($)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()