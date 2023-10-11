# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:01:21 2023

@author: joaomateus
"""

import numpy as np



def sigmoid(soma):
    
    '''formula da função de ativição sigmoid'''
    return 1/(1 + np.exp(-soma))


def sigmoidDerivada(sig):
    return sig*(1-sig)


a = sigmoid(0.5)


aDerivada = sigmoidDerivada(a)


entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])



#pesos0 = np.array([[-0.424, -0.740, -0.961],
#                   [0.358, -0.577, -0.469]])

pesos0 = 2*np.random.random((2,3))-1



#pesos1 = np.array([[-0.017], 
#                   [-0.893], 
#                   [0.148]])

pesos1 = 2*np.random.random((3,1))-1




'''quantidade de vezes que vai rodar para arrumar os pesos'''
epocas = 1000000
taxaAprendizagem = 0.6
momento = 1


for j in range(epocas):
    
    camadaEntrada = entradas
    
    somaSinapse0 = np.dot(camadaEntrada, pesos0)  
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    