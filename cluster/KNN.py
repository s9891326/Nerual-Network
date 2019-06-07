# -*- coding: utf-8 -*-
"""
Created on Sun May 19 22:05:18 2019

@author: eddy
"""

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("section: k-nearest neighbors")
iris = datasets.load_iris()
x = iris.data
y = iris.target
print("class labels :", np.unique(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)
print("x_test = ", x_test.size)
print("y_test = ", y_test.size)

#歐式距離
k_range = range(1,26)
scores_list = []

#for i in k_range:
#    knn = KNeighborsClassifier(n_neighbors = i)
#    knn.fit(x_train, y_train)
#
#    y_pred = knn.predict(x_test)
#    scores_list.append(metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

classes = {0:'setosa', 1:'versicolor', 2:'virginica'}
y_predict = knn.predict(x_test)

print("y_predict = " ,y_predict)
print("y_test = ", y_test)

X = np.arange(30)
plt.scatter(X, y_test, color = 'red')
plt.scatter(X, y_predict, color = 'blue')
plt.xlabel("number")
plt.ylabel("flower cluster")
plt.show()

#plt.plot(k_range, scores_list)
#plt.xlabel('value of k for knn')
#plt.ylabel('testing accuracy')

