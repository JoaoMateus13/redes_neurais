# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:56:10 2023

@author: joaomateus
"""

import numpy as np


'''Uma matriz de entrada A(0 0
                           0 1
                           1 0  L X C
                           1 1) 4x2'''
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])



'''Uma matriz de saida   B(0
                           0
                           1
                           1) 4x1'''


'''Saida de uma porta OR'''
saidas = np.array([0,
                  1,
                  1,
                  1])


'''Matriz de entrada contem 2 colunas, logo são 2 pesos'''
pesos = np.array([0.0,
                  0.0])


taxaAprendizagem = 0.1 



def stepFunction(soma):
    if(soma >= 1):
        return 1
    
    return 0


def calculaSaida(registro):
    '''produto escalar da entrada pelo peso'''
    s = registro.dot(pesos)
    
    return stepFunction(s)


def treinar():
    erroTotal = 1
    
    while(erroTotal != 0):  
        erroTotal = 0
        for i in range(len(saidas)):
            '''np.asarray(entradas[1]) = pega a linha da matriz entrada'''
            saidaCalculado = calculaSaida(np.asarray(entradas[i]))
            
            '''abs = modolo |erro|'''
            erro = abs(saidas[i] - saidaCalculado)
            
            erroTotal += erro
            
            for j in  range(len(pesos)):
                pesos[j] = pesos[j] +  (taxaAprendizagem * entradas[i][j] * erro)
                
                print('Pesos atualizado: '+ str(pesos[j]))
                
                
        print('total de erros: ' + str(erroTotal))
        
        
        

treinar()


print(calculaSaida(np.array([0,1])))
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        