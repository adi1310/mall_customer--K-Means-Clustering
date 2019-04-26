# K-Means Clustering

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Datset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

# Using the Elbow Method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300,
                    n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabe('WCSS')
plt.show()

# Applying K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300,
                    n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

# Visualising the Clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'red',     label = 'careful')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'blue',    label = 'standard')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green',   label = 'target')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'cyan',    label = 'careless')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'sensable')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids')
plt.title('clustet of clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 





   