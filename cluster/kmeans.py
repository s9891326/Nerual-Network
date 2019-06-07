# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:38:02 2019

@author: eddy
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

np.random.seed(800)

nb_samples = 800

def show_dataset(X):
    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    
    ax.grid()
    ax.set_xlabel('salary')
    ax.set_ylabel('age')
    
    ax.scatter(X[:, 0], X[:, 1], marker = 'o', color = 'b')
    plt.show()

def show_cluster_dataset(X, kmeanCluster):
    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    
    ax.grid()
    ax.set_xlabel('salary')
    ax.set_ylabel('age')
    
    for i in range(nb_samples):
        c = kmeanCluster.predict(X[i].reshape(1, -1))
        if c == 0:
            ax.scatter(X[i, 0], X[i, 1], marker = 'o', color = 'r')
        elif c == 1:
            ax.scatter(X[i, 0], X[i, 1], marker = '^', color = 'g')
        else:
            ax.scatter(X[i, 0], X[i, 1], marker = 'd', color = 'b')
        
if __name__ == '__main__':
    X, Y = make_blobs(n_samples = nb_samples, n_features = 2, centers = 3, cluster_std = 0.5)
    print(X)
    
    show_dataset(X)
    
    KmeanCluster = KMeans(n_clusters = 3)
    KmeanCluster.fit(X)
    
    print("kmeans cluster = ", KmeanCluster.cluster_centers_)
    
    show_cluster_dataset(X, KmeanCluster)
    