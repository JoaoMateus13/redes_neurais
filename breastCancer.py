# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 00:48:45 2023

@author: joaom
"""

import numpy as np
from sklearn import datasets




def sigmoid(soma):
    
    '''formula da função de ativição sigmoid'''
    return 1/(1 + np.exp(-soma))


def sigmoidDerivada(sig):
    return sig*(1-sig)





base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target
saidas = np.empty([569,1], dtype = int)

for i in range(len(valoresSaida)):
    saidas[i] = valoresSaida[i]



pesos0 = 2*np.random.random((30,15))-1

pesos1 = 2*np.random.random((15,1))-1




'''quantidade de vezes que vai rodar para arrumar os pesos'''
epocas = 100000
taxaAprendizagem = 0.2
momento = 1

 
for j in range(epocas):
    
    camadaEntrada = entradas
    
    somaSinapse0 = np.dot(camadaEntrada, pesos0)  
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    "Formas de calcular o erro"
    erroCamadaSaida = saidas - camadaSaida
    MSE = np.mean((saidas - camadaSaida)**2)
    RMSE = np.sqrt(np.mean((saidas - camadaSaida)**2))
    
    mediaAbsoluta = np.mean(abs(erroCamadaSaida))
    
    print("Erro: " + str(mediaAbsoluta))
    
    
    "Calculo do delta"
    derivadaSig = sigmoidDerivada(camadaSaida) 
    deltaSaida = erroCamadaSaida * derivadaSig
    
    
    "transformar pesos1 para sua transposta para realizar a multiplicacao"
    "Calculo do delta"
    pesos1Transposta = pesos1.T
    deltaSaidaXPesos1 = deltaSaida.dot(pesos1Transposta)
    
    deltaCamadaOculta =  sigmoidDerivada(camadaOculta) *deltaSaidaXPesos1
    
    "backpropagation"
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1*momento) + (pesosNovo1 * taxaAprendizagem)
    
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    