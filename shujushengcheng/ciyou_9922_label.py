import numpy as np
import pandas as pd
import math



x = pd.read_csv(r'/home/wwj/chenqiliang/select9922/9922data.csv', header=None)
matrix = x.values
def GetLabel(i1, i2, j1, j2):
    for i in range(0, 1296):
        if i1 == matrix[i, 1] and i2 == matrix[i, 2] and j1 == matrix[i, 3] and j2 == matrix[i,4]:
            return matrix[i][0]


def maxNormIndex(A):
    max_Gain = 0
    max_i = 0
    max_j = 0
    for i in range(0, 9):
        for j in range(0, 9):
           
            G = math.sqrt(A[i,j]*A[i,j])
            if G > max_Gain:
               max_Gain = G
               max_i = i
               max_j = j
  
    return [max_i, max_j]

def JTRAS_Capacity_Algorithm_8822(A1, p):
    B = np.mat(np.zeros((2, 2)), dtype=float)
    transmit = {0, 1, 2, 3, 4, 5, 6, 7,8}
    receive = {0, 1, 2, 3, 4, 5, 6, 7,8}
   
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])
    
    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}

    max_C = 0
    add_transmit  = 0
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

    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_C,selected_transmit_list[0],selected_transmit_list[1],selected_receive_list[0],selected_receive_list[1]
def JTRAS_Gain_Algorithm_8822(A1):
    transmit = {0, 1, 2, 3, 4, 5, 6, 7,8}
    receive = {0, 1, 2, 3, 4, 5, 6, 7,8}
  
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])
 
    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}

    max_G = 0
    add_transmit = 0
    add_receive = 0
    for i in transmit:   
        for j in receive: 
            B = np.mat(np.zeros((2, 2)), dtype=float)
         
            B[0,0] = A1[maxNormIndex(A1)[0],maxNormIndex(A1)[1]]
           
            B[0,1] = A1[maxNormIndex(A1)[0],j]
          
            B[1,0] = A1[i,maxNormIndex(A1)[1]]
            B[1,1] = A1[i,j]
            G = math.sqrt(1 / 2) * np.linalg.norm(B, ord='fro')
            if G > max_G:
                max_G = G
                add_transmit = i
                add_receive = j

    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_G,selected_transmit_list[0],selected_transmit_list[1],selected_receive_list[0],selected_receive_list[1]












dataset = pd.read_csv(open(r'/home/wwj/chenqiliang/101022data/92All-channel_matrix_p_10.csv')).iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 9, 9)




length = 200000
p = 20  


subCapacity_List = []
subGain_List = []

capacityLabel_List = []
gainLabel_List = []


for i in range(0,200000):
    print(i)
    subCapacity,i1_Capacity,i2_Capacity,j1_Capacity,j2_Capacity = JTRAS_Capacity_Algorithm_8822(dataset[i],p)  
    subGain, i1_Gain, i2_Gain, j1_Gain, j2_Gain = JTRAS_Gain_Algorithm_8822(dataset[i])  
                                                                                                  

    subCapacity_List.append(subCapacity)
    subGain_List.append(subGain)
    capacityLabel_List.append(GetLabel(i1_Capacity,i2_Capacity,j1_Capacity,j2_Capacity))
    gainLabel_List.append(GetLabel(i1_Gain, i2_Gain, j1_Gain, j2_Gain))



subCapacityData = pd.DataFrame(subCapacity_List)

capacityLabelData = pd.DataFrame(capacityLabel_List)


subGainData = pd.DataFrame(subGain_List)

gainLabelData = pd.DataFrame(gainLabel_List)


#subCapacityData.to_csv(r'/home/wwj/chenqiliang/101022data/92JTRAS_subcapacity_50.csv')
capacityLabelData.to_csv(r'/home/wwj/chenqiliang/101022data/92JTRAS_capacity_label_20.csv')
#subGainData.to_csv(r'/home/wwj/chenqiliang/101022data/92JTRAS_subgain_50.csv')
gainLabelData.to_csv(r'/home/wwj/chenqiliang/101022data/92JTRAS_gainlabel_20.csv')
