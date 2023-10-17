# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:05:59 2023

@author: joaom
"""

from sklearn.neural_network import MLPClassifier
from sklearn import datasets


iris = datasets.load_iris()


entradas = iris.data

saidas = iris.target


redeNeural = MLPClassifier(activation='logistic',verbose=True,
                           max_iter=10000,
                           tol=0.000001,
                           learning_rate_init=0.001)

redeNeural.fit(entradas, saidas)




redeNeural.predict([[5,1, 2, 2]])







