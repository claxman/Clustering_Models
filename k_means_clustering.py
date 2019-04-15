#Author: Chaitanya Laxman

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("No. of clusters")
plt.ylabel("WCSS")
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_pred == 0,0], X[y_pred == 0,1], s = 40, c = "green", label = "cluster1")
plt.scatter(X[y_pred == 1,0], X[y_pred == 1,1], s = 40, c = "blue", label = "cluster2")
plt.scatter(X[y_pred == 2,0], X[y_pred == 2,1], s = 40, c = "grey", label = "cluster3")
plt.scatter(X[y_pred == 3,0], X[y_pred == 3,1], s = 40, c = "purple", label = "cluster4")
plt.scatter(X[y_pred == 4,0], X[y_pred == 4,1], s = 40, c = "black", label = "cluster5")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 100, c = "red", label = "centroid")

plt.title("Cluster of customers")
plt.xlabel("Annual Income($)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()