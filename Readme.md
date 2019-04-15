# Different Clustering Models!

  - K-Means Clustering
  - Hierarchical Clustering

# Most Asked Question
#### How do I select optimal number of clusters?
###### For **K-Means**, we use ***The Elbow Method*** in order to determine the optimal number of clusters.
We calculate ***Within-Cluster-Sum-of-Squares(WCSS)*** 

![](https://raw.githubusercontent.com/claxman/Clustering_Models/master/images/CodeCogsEqn.png)

P - Every point in cluster-1
C - Centroid
and so on...
Let's get WCSS values:
```python
from sklearn.cluster import KMeans
wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusterss = i, init = "k-means++", random_state = 0)
    kmeans.fit(X) # X is a dataframe containing important features.  
    wcss.append(kmeans.inertia_)
```

And after plotting WCSS against No. of clusters, we get this plot.

![](https://raw.githubusercontent.com/claxman/Clustering_Models/master/images/Elbow_Method.png)

We can see that after n = 5, WCSS actually started to decrease frequently.

###### For **Hierarchical Clustering** especially for ***Agglomerative Clustering***, we use ***Dendrograms*** in order to determine the optimal number of clusters.

![](https://raw.githubusercontent.com/claxman/Clustering_Models/master/images/dendrogram.png)
- We look for the longest verticle line that doesn't cross any extended horizontal lines.
- Then we draw a horizontal line and set it as a thershold according to the previous rule.
- Finally, we count the crossing points.
- For example. we see from 248 - 101 we are getting longest verticle line.
- Hence, we draw a horizontal line within this range.
- That horizontal line will cut 5 verticle lines giving us 5 cluster points.
    
**Author**
------
#### Chaitanya Laxman
