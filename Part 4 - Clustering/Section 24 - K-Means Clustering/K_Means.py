#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:26:49 2018

@author: sid
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss Value')
plt.show() 
#Hence we concluded by elbow method that the number of clusters is 5

#Applying K_Means to the null dataset
k_means = KMeans(n_clusters = 5, init='k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_k_means = k_means.fit_predict(X)
    
#Visualising the clusters
plt.scatter(X[y_k_means == 0,0], X[y_k_means == 0,1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_k_means == 1,0], X[y_k_means == 1,1], s=100, c='blue', label='Cluster 2')                    
plt.scatter(X[y_k_means == 2,0], X[y_k_means == 2,1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_k_means == 3,0], X[y_k_means == 3,1], s=100, c='pink', label='Cluster 4')
plt.scatter(X[y_k_means == 4,0], X[y_k_means == 4,1], s=100, c='orange', label='Cluster 5')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Cluster of Customers')
plt.xlabel('Annual Income ($) in k')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()