# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:38:51 2019

@author: eddy
"""

from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X,y = make_classification(n_samples = 300, n_features=2, n_redundant=0, n_informative=2,random_state=3,scale=100,n_clusters_per_class=1)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

X = preprocessing.scale(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = SVC()
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))
