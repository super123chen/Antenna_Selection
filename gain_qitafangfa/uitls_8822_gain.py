import math

import numpy as np


def decoupledSelection_8822(data):
    max_transmit = 0
    T1 = 0
    T2 = 0
    G = 0

    for i in range(0, 8):
        for j in range(i + 1, 8):
            B = data[[i, j], :]
            new_G = math.sqrt(1 / 2) * np.linalg.norm(B, ord='fro')
            if new_G > max_transmit:
                max_transmit = new_G
                T1 = i
                T2 = j

    for i in range(0, 8):
        for j in range(i + 1, 8):
            B = data[[T1, T2]][:, [i, j]]
            new_G = math.sqrt(1 / 2) * np.linalg.norm(B, ord='fro')
            if new_G > G:
                G = new_G
    return G




def maxChannelGain_8822(A):
    G_new = 0.       
    Count_new = 0  
    Count = 0
    
    for i1 in range(0, 8):
        for i2 in range(i1+1, 8):
            
            for j1 in range(0, 8):
                for j2 in range(j1+1, 8):
                
                    if i1 != i2:
                        if j1 != j2:
                            B = A[[i1, i2]][:, [j1, j2]]
                            G = math.sqrt(1/2) * np.linalg.norm(B, ord='fro')
                           
                            Count = Count + 1
                            if G > G_new:
                               G_new = G
                               Count_new = Count
    
    return [G_new, Count_new]



def computation_time(test):
    if test < 1e-6:
        testUnit = "ns"
        test *= 1e9
    elif test < 1e-3:
        testUnit = "us"
        test *= 1e6
    elif test < 1:
        testUnit = "ms"
        test *= 1e3
    else:
        testUnit = "s"
    return [test,testUnit]