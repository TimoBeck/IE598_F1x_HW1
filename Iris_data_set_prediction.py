#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:27:26 2019

@author: timotheebecker
"""
# This code is from Python machine learning : machine learning...
#Import of packages
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt

#
from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
print (X_iris.shape, y_iris.shape)
print (X_iris[0], y_iris[0])

#Training data set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Get dataset with only the first two attributes
X, y = X_iris[:, :2], y_iris

# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
print (X_train.shape, y_train.shape)

# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
     xs = X_train[:, 0][y_train == i]
     ys = X_train[:, 1][y_train == i]
     plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train) 

print (clf.coef_)
print (clf.intercept_)

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i]-xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(xs, ys)

print (clf.predict(scaler.transform([[4.7, 3.1]])))

print (clf.decision_function(scaler.transform([[4.7, 3.1]])))


from sklearn import metrics
y_train_pred = clf.predict(X_train)
print (metrics.accuracy_score(y_train, y_train_pred))

y_pred = clf.predict(X_test)
print (metrics.accuracy_score(y_test, y_pred))

print (metrics.classification_report(y_test, y_pred,target_names=iris.target_names))

print (metrics.confusion_matrix(y_test, y_pred))

from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# create a composite estimator made by a pipeline of the standarization and the linear model
clf = Pipeline([('scaler', StandardScaler()),('linear_model', SGDClassifier())])
# create a k-fold cross validation iterator of k=5 folds

cv = KFold(5, True, 33) # ---------- modified, removed X.shape[0]
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf, X, y, cv=cv)
print (scores)

from scipy.stats import sem
def mean_score(scores):
 return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

print (mean_score(scores))

print("My name is Timothee Becker")
print("My NetID is: tbecker5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
