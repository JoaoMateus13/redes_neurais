# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:01:21 2023

@author: joaomateus
"""

import numpy as np



def sigmoid(soma):
    
    '''formula da função de ativição sigmoid'''
    return 1/(1 + np.exp(-soma))



entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])



pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])



pesso1 = np.array([[-0.017], 
                   [-0.893], 
                   [0.148]])


'''quantidade de vezes que vai rodar para arrumar os pesos'''
epocas = 100


for j in range(epocas):
    
    camadaEntrada = entradas
    
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSaida = np.dot(camadaOculta, pesso1)
    
    camadaSaida = sigmoid(somaSaida)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    