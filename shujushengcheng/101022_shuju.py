from numpy import mat
import math
import pandas as pd
import time
import numpy as np



It = np.eye(2)
p = 20           
m = n = 10      
Nt = 2           



def maxChannelCapacity(A):
    C_new = 0
    Count_new = 0
    Count = 0
   
    for i1 in range(0, 10):
        for i2 in range(i1 + 1, 10):
            
            for j1 in range(0, 10):
                for j2 in range(j1 + 1, 10):
                
                    if i1 != i2:
                        if j1 != j2:
                            B = A[[i1, i2]][:, [j1, j2]]
                            
                            
                            C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
                            
                            Count = Count + 1
                            if C > C_new:  
                                C_new = C
                                Count_new = Count
  
    return [C_new, Count_new]




def maxChannelGain(A):
    G_new = 0.      
    Count_new = 0   
    Count = 0
  
    for i1 in range(0, 10):
        for i2 in range(i1+1, 10):
            
            for j1 in range(0, 10):
                for j2 in range(j1+1, 10):
                    
                    if i1 != i2:
                        if j1 != j2:
                            B = A[[i1, i2]][:, [j1, j2]]
                            
                            G = math.sqrt(1/2) * np.linalg.norm(B, ord='fro')
                           
                            Count = Count + 1
                            if G > G_new:
                               G_new = G
                               Count_new = Count
    
    return [G_new, Count_new]

matrix_List= []                       
fullCapacity_List = []                
subCapacity_List = []                 
fullGain_List = []                    
subGain_List = []                    
capacityLabel_List = []               
gainLabel_List = []                 



length = 200000
I = np.eye(10)

for i in range(0,length):
    A = math.sqrt(1.0/2) * (np.random.rand(m,n)+ 1j*np.random.rand(m,n))
    
    A = np.matrix(abs(A))

    
    nor1 = np.full(A.shape,np.max(A) - np.min(A))          
    A1 = A - np.full(A.shape,np.min(A))
   
    A1 = np.divide(A1,nor1)

    
    t = np.array(A1).reshape(1, -1)  
    matrix_List.append(t[0])

   
    fullCapacity = np.log2(np.linalg.det(I + p * A1.T * A1 / n))
    fullCapacity_List.append(fullCapacity)
   
    fullGain = math.sqrt(1 / 2) * np.linalg.norm(A1, ord='fro')
    fullGain_List.append(fullGain)

    
    R1 = maxChannelCapacity(A1)
    R2 = maxChannelGain(A1)
   
    subCapacity = R1[0]
    capacityLabel = R1[1]
    subGain = R2[0]
    gainLabel = R2[1]

    subCapacity_List.append(subCapacity)
    capacityLabel_List.append(capacityLabel)

    subGain_List.append(subGain)
    gainLabel_List.append(gainLabel)

    
    if i % 1000 == 0:
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(i)


matrixData = pd.DataFrame(matrix_List)

fullCapacityData = pd.DataFrame(fullCapacity_List)

fullGainData = pd.DataFrame(fullGain_List)

subCapacityData = pd.DataFrame(subCapacity_List)

subGainData = pd.DataFrame(subGain_List)

capacityLabelData = pd.DataFrame(capacityLabel_List)

gainLabelData = pd.DataFrame(gainLabel_List)


matrixData.to_csv(r'/home/wwj/chenqiliang/101022data/102All-channel_matrix_p_20.csv')
#fullCapacityData.to_csv(r'/home/wwj/chenqiliang/101022data/102gain_labels_p_50.csv')
#subCapacityData.to_csv(r'/home/wwj/chenqiliang/101022data/102Sub_channel_capacity_p_50.csv')

#fullGainData.to_csv(r'/home/wwj/chenqiliang/101022data/102All_channel_gain_p_50.csv')
#subGainData.to_csv(r'/home/wwj/chenqiliang/101022data/102Sub_channel gain_p_50.csv')

#capacityLabelData.to_csv(r'/home/wwj/chenqiliang/101022data/102capacity_labels_p_50.csv')
#gainLabelData.to_csv(r'/home/wwj/chenqiliang/101022data/102gain_labels_p_50.csv')