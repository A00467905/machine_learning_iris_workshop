#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: amangahir
"""

import streamlit as st
from joblib import load
from sklearn import tree

st.title("IRIS Prediction")
lables = ['setosa', 'versicolor', 'virginica']

ml_clf = load("ML_DT.joblib")

sepal_length = st.slider('sepal length (cm)', min_value=0, max_value=10)
sepal_width = st.slider('sepal width (cm)', min_value=0, max_value=10)
petal_length = st.slider('petal length (cm)', min_value=0, max_value=10)
petal_width = st.slider('petal width (cm)', min_value=0, max_value=10)

prediction = ml_clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

st.write(lables[prediction[0]])