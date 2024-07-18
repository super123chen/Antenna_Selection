import numpy as np
import pandas as pd
import math

x = pd.read_csv(r'/home/wwj/chenqiliang/ceshiyongde/8844select.csv', header=None)
matrix = x.values
p=10
def GetLabel(i1, i2, i3, i4, j1, j2, j3, j4):
    for i in range(0, 4900):
        if i1 == matrix[i, 1] and i2 == matrix[i, 2] and i3 == matrix[i, 3] and i4 == matrix[i, 4] and j1 == matrix[
            i, 5] and j2 == matrix[i, 6] and j3 == matrix[i, 7] and j4 == matrix[i, 8]:
            return matrix[i][0]


def maxNormIndex(A):
    max_C = 0
    max_i = 0
    max_j = 0
    for i in range(0, 8):
        for j in range(0, 8):

            C = math.sqrt(A[i, j] * A[i, j])
            if C > max_C:
                max_C = C
                max_i = i
                max_j = j

    return [max_i, max_j]


def JTRAS_C_Algorithm_8844(A1):
    transmit = {0, 1, 2, 3, 4, 5, 6, 7}
    receive = {0, 1, 2, 3, 4, 5, 6, 7}
    transmit.remove(maxNormIndex(A1)[0])
    receive.remove(maxNormIndex(A1)[1])

    selected_transmit = {maxNormIndex(A1)[0]}
    selected_receive = {maxNormIndex(A1)[1]}

    max_C = 0
    add_transmit = 0
    add_receive = 0
    I2 = np.eye(2)
    for i in transmit:
        for j in receive:
            B = np.mat(np.zeros((2, 2)), dtype=float)

            B[0, 0] = A1[maxNormIndex(A1)[0], maxNormIndex(A1)[1]]

            B[0, 1] = A1[maxNormIndex(A1)[0], j]

            B[1, 0] = A1[i, maxNormIndex(A1)[1]]
            B[1, 1] = A1[i, j]
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
    I2 = np.eye(2)
    for i in transmit:
        for j in receive:
            B = np.mat(np.zeros((2, 2)), dtype=float)

            B[0, 0] = A1[maxNormIndex(A1)[0], maxNormIndex(A1)[1]]

            B[0, 1] = A1[maxNormIndex(A1)[0], j]

            B[1, 0] = A1[i, maxNormIndex(A1)[1]]
            B[1, 1] = A1[i, j]
            C = np.log2(np.linalg.det(I2 + p * B.T * B / 2))
            if C > max_C:
                max_C = C
                add_transmit = i
                add_receive = j
    transmit.remove(add_transmit)
    receive.remove(add_receive)
    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)
    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    max_C = 0
    add_transmit = 0
    add_receive = 0
    I2 = np.eye(2)
    for i in transmit:
        for j in receive:
            B = np.mat(np.zeros((2, 2)), dtype=float)
            B[0, 0] = A1[selected_transmit_list[0], selected_receive_list[0]]
            B[0, 1] = A1[selected_transmit_list[0], j]
            B[1, 0] = A1[i, selected_receive_list[0]]
            B[1, 1] = A1[i, j]
            C = np.log2(np.linalg.det(I2 + p * B.T * B / 2))
            if C > max_C:
                max_C = C
                add_transmit = i
                add_receive = j

    # transmit.remove(add_transmit)
    # receive.remove(add_receive)
    selected_transmit.add(add_transmit)
    selected_receive.add(add_receive)

    selected_transmit_list = sorted(selected_transmit)
    selected_receive_list = sorted(selected_receive)

    return max_C, selected_transmit_list[0], selected_transmit_list[1], selected_transmit_list[2], \
           selected_transmit_list[3], selected_receive_list[0], selected_receive_list[1], selected_receive_list[2], \
           selected_receive_list[3]





dataset = pd.read_csv(open(r'/home/wwj/chenqiliang/ceshiyongde/208833All-channel_matrix_p_10.csv')).iloc[:, 1:]

dataset = np.asarray(dataset, np.float32)

dataset = dataset.reshape(dataset.shape[0], 8, 8)

length = 200000

subGain_List = []
gainLabel_List = []
gainLabel_info = []

for i in range(0, 200000):
    list = []
    print(i)
    subGain, i1_C, i2_C, i3_C, i4_C, j1_C, j2_C, j3_C, j4_C = JTRAS_C_Algorithm_8844(
        dataset[i])
    subGain_List.append(subGain)
    #list = [i1_Gain, i2_Gain, i3_Gain, i4_Gain, j1_Gain, j2_Gain, j3_Gain, j4_Gain]
    #list = [i1_C, i2_C, i3_C,i4_C ,j1_C, j2_C, j3_C,j4_C]
    #if i2_C>i1_C and i3_C>i2_C and i4_C>i3_C and j2_C>j1_C and j3_C>j2_C and j4_C>i3_C:
    gainLabel_List.append(GetLabel(i1_C, i2_C, i3_C, i4_C, j1_C, j2_C, j3_C, j4_C))

subGainData = pd.DataFrame(subGain_List)

gainLabelData = pd.DataFrame(gainLabel_List)

subGainData.to_csv(r'/home/wwj/chenqiliang/8833and8844shuju/8844JTRAS_subcapacity_10.csv')
gainLabelData.to_csv(r'/home/wwj/chenqiliang/8833and8844shuju/8844JTRAS_capacity_label_10.csv')

# In[ ]:



