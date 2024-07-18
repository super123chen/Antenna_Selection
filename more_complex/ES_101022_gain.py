from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Best_subGain = pd.read_csv(r"/home/wwj/chenqiliang/101022data/102Sub_channel gain_p_10.csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:200000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.1, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('SNR=10')
print(Best_subGain_Mean)







Best_subGain = pd.read_csv(r"/home/wwj/chenqiliang/101022data/102Sub_channel gain_p_20.csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:200000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.1, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('SNR=20')
print(Best_subGain_Mean)






Best_subGain = pd.read_csv(r"/home/wwj/chenqiliang/101022data/102Sub_channel gain_p_30.csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:200000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.1, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('SNR=30')
print(Best_subGain_Mean)




Best_subGain = pd.read_csv(r"/home/wwj/chenqiliang/101022data/102Sub_channel gain_p_40.csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:200000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.1, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('SNR=40')
print(Best_subGain_Mean)




Best_subGain = pd.read_csv(r"/home/wwj/chenqiliang/101022data/102Sub_channel gain_p_50.csv").iloc[:, 1:]
Best_subGain = Best_subGain[0:200000]
Best_subGain = np.asarray(Best_subGain, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subGain, Best_subGain, test_size=0.1, random_state=40)
Best_subGain_Mean = np.mean(yTest)
print('SNR=50')
print(Best_subGain_Mean)

