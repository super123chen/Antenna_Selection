

import numpy as np
import pandas as pd
import math

p=10
x = pd.read_csv(r'/home/wwj/chenqiliang/ceshiyongde/8833select.csv', header=None)
matrix = x.values

def GetLabel(i1, i2, i3, j1, j2, j3):
    for i in range(0, 3136):
        if i1 == matrix[i, 1] and i2 == matrix[i, 2] and i3 == matrix[i,3] and j1 == matrix[i, 4] and j2 == matrix[i,5] and j3 == matrix[i,6]:
            return matrix[i][0]


def maxNormIndex(A):
    max_Gain = 0
    max_i = 0
    max_j = 0
    for i in range(0, 8):
        for j in range(0, 8):
            G = math.sqrt(A[i,j]*A[i,j])
            if G > max_Gain:
               max_Gain = G
               max_i = i
               max_j = j
    return [max_i, max_j]

def JTRAS_Capacity_Algorithm_8833(A1,p):
    B = np.mat(np.zeros((2, 2)), dtype=float)
    transmit = {0, 1, 2, 3, 4, 5, 6, 7}
    receive = {0, 1, 2, 3, 4, 5, 6, 7}
 
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])
  
    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}
    p=10
    max_C = 0
    add_transmit = 0
    add_receive = 0
    I2 = np.eye(2)
    
    for i in transmit:   
        for j in receive:
            
            B[0,0] = A1[maxNormIndex(A1)[0],maxNormIndex(A1)[1]]
            
            B[0,1] = A1[maxNormIndex(A1)[0],j]
            
            B[1,0] = A1[i,maxNormIndex(A1)[1]]
            B[1,1] = A1[i,j]
            C = np.log2(np.linalg.det(I2 + p * B.T * B / 2))
            if C > max_C:
                max_C = C
                add_transmit = i
                add_receive = j
    
    
    transmit.remove(add_transmit)
    receive.remove(add_receive)
    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    
    max_C = 0
    add_transmit = 0
    add_receive = 0
    for i in transmit:   
        for j in receive: 
            B = np.mat(np.zeros((2, 2)), dtype=float)
          
            B[0,0] = A1[maxNormIndex(A1)[0],maxNormIndex(A1)[1]]
           
            B[0,1] = A1[maxNormIndex(A1)[0],j]
           
            B[1,0] = A1[i,maxNormIndex(A1)[1]]
            B[1,1] = A1[i,j]
            C = np.log2(np.linalg.det(I2 + p * B.T * B / 2))
            if C > max_C:
                max_C = C
                add_transmit = i
                add_receive = j
    
    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    
    
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_C,selected_transmit_list[0],selected_transmit_list[1],selected_transmit_list[2],selected_receive_list[0],selected_receive_list[1],selected_receive_list[2]

dataset = pd.read_csv(open(r'/home/wwj/chenqiliang/ceshiyongde/208833All-channel_matrix_p_10.csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8)

length = 200000
p = 10  

subCapacity_List = []
capacityLabel_List = []
for i in range(0,200000):
    print(i)
    subCapacity,i1_Capacity,i2_Capacity,i3_Capacity,j1_Capacity,j2_Capacity,j3_Capacity = JTRAS_Capacity_Algorithm_8833(dataset[i],p)
    subCapacity_List.append(subCapacity)
    capacityLabel_List.append(GetLabel(i1_Capacity,i2_Capacity,i3_Capacity,j1_Capacity,j2_Capacity,j3_Capacity))


subCapacityData = pd.DataFrame(subCapacity_List)

capacityLabelData = pd.DataFrame(capacityLabel_List)

subCapacityData.to_csv(r'/home/wwj/chenqiliang/8833and8844shuju/8833JTRAS_subcapacity_10.csv')
capacityLabelData.to_csv(r'/home/wwj/chenqiliang/8833and8844shuju/8833JTRAS_capacity_label_10.csv')

# In[ ]:




