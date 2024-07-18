import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from uitls_8822_capacity import decoupledSelection_8822,maxChannelCapacity_8822, computation_time


a = 10


dataset = pd.read_csv(open(r"/home/wwj/chenqiliang/101022data/92All-channel_matrix_p_10.csv")).iloc[:, 1:]

dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)

label = pd.read_csv(open(r"/home/wwj/chenqiliang/101022data/92capacity_labels_p_10.csv")).iloc[:, 1]

label = np.asarray(label, np.int32)
label.astype(np.int32)

n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))  
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1  

xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
xTest_np = np.array(xTest[0:40000])



I1 = np.eye(8)
I2 = np.eye(2)

Pre_Loss = []

Pre_Capacity = []

testTime = []
for i in range(40000):
    print(i)
  
    ArrayA = xTest_np[i].reshape(8, 8)
    
    ArrayA = np.matrix(ArrayA)
    

    testStart = time()
    Pre_subCapacity = decoupledSelection_8822(ArrayA,a)
    test = time() - testStart
   
    testTime.append(test)


    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)





Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)



print("SNR",a)

#print((computation_time(sum(testTime))[0], computation_time(sum(testTime))[1]))
print('subcapacitymean', Capacity_Mean)
print('loss', Loss_Mean)
print('lossvar', Loss_Variance)

