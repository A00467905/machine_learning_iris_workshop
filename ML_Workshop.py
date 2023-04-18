#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:10:27 2023

@author: amangahir
"""
from joblib import dump
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris_data = load_iris()

X = iris_data['data']
y = iris_data['target']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=44)

clf = tree.DecisionTreeClassifier()
clf.fit(train_X,train_y)

y_prediction_results = clf.predict(test_X)

accuracy_metric = accuracy_score(test_y, y_prediction_results)

print(accuracy_metric)

print(iris_data['target_names'][y_prediction_results])

dump(clf, "ML_DT.joblib")