#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 02:46:46 2018

@author: sid
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using Dendrogram to find the optimal Clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
plt.show() 

#Fitting the dataset into the model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#Visualising the clusters
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s=100, c='blue', label='Cluster 2')                    
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s=100, c='pink', label='Cluster 4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s=100, c='orange', label='Cluster 5')
plt.title('Cluster of Customers')
plt.xlabel('Annual Income ($) in k')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
