from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Best_subCapacity = pd.read_csv(r"/home/wwj/chenqiliang/ceshiyongde/208833Sub_channel_capacity_p_10.csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:100000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('SBR10')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"/home/wwj/chenqiliang/ceshiyongde/208833Sub_channel_capacity_p_20.csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:100000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('SNR20')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"/home/wwj/chenqiliang/ceshiyongde/208833Sub_channel_capacity_p_30.csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:100000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('SNR30')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"/home/wwj/chenqiliang/ceshiyongde/208833Sub_channel_capacity_p_30.csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:100000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('SNR40')
print(Best_subCapacity_Mean)

Best_subCapacity = pd.read_csv(r"/home/wwj/chenqiliang/ceshiyongde/208833Sub_channel_capacity_p_50.csv").iloc[:, 1:]
Best_subCapacity = Best_subCapacity[0:100000]
Best_subCapacity = np.asarray(Best_subCapacity, np.float32)
xTrain, xTest, yTrain, yTest = train_test_split(Best_subCapacity, Best_subCapacity, test_size=0.2, random_state=40)
Best_subCapacity_Mean = np.mean(yTest)
print('SNR50')
print(Best_subCapacity_Mean)