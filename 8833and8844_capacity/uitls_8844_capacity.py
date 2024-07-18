
import numpy as np
def decoupledSelection_8844(data,p):
    I = np.eye(8)
    It = np.eye(4)
    Nt = 4
    max_transmit = 0
    T1 = 0
    T2 = 0
    T3 = 0
    T4 = 0
    C = 0

   
    for i in range(0, 8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                for l in range(k + 1, 8):
            
                   B = data[[i, j, k,l], :]
                   new_C = np.linalg.det(I + p * B.T * B / Nt)
                   if new_C > max_transmit:
                     max_transmit = new_C
                     T1 = i
                     T2 = j
                     T3 = k
                     T4 = l

    for i in range(0, 8):
        for j in range(i + 1, 8):
           for k in range(j + 1, 8):
              for l in range(k + 1, 8):
                  B = data[[T1, T2, T3, T4]][:, [i, j,k,l]]
                  new_C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
                  if new_C > C:
                     C = new_C
    return C


def maxChannelCapacity_8844(A,p):
    C_new = 0
    Count_new = 0
    Count = 0
    Nt = 4
    It = np.eye(4)

    for i1 in range(0, 8):
        for i2 in range(i1+1, 8):
           for i3 in range(i2+1, 8):
              for i4 in range(i3+1, 8):
                 for j1 in range(0, 8):
                    for j2 in range(j1+1, 8):
                       for j3 in range(j2+1, 8):
                          for j4 in range(j3+1, 8):
                              B = A[[i1, i2, i3,i4]][:, [j1, j2, j3,j4]]
                            
                              C = np.log2(np.linalg.det(It + p * B.T * B / Nt))
                            
                              Count = Count + 1
                              if C > C_new:    
                                 C_new = C
                                 Count_new = Count
    
    return [C_new,Count_new]




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