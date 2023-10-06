# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 9:16:10 2023

@author: joaomateus
"""
import numpy as np


entradas = np.array([1,7,8])

pesos =  np.array([0.8, 0.1, 0])



def soma(e, p):
    s = 0
    
    '''Realiza um produto escalar dos vetores '''
    return e.dot(p)
        
        
s = soma(entradas, pesos)


def stepFunction(soma):
    if(soma >= 1):
        return 1
    
    return 0


t = stepFunction(s)
