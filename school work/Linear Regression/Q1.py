# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_train1 = pd.read_csv('./train-100-10.csv')
data_test1 = pd.read_csv('./test-100-10.csv')
data_train2 = pd.read_csv('./train-100-100.csv')
data_test2 = pd.read_csv('./test-100-100.csv')
data_train3 = pd.read_csv('./train-1000-100.csv')
data_test3 = pd.read_csv('./test-1000-100.csv')
data_train4 = pd.read_csv('./train-50(1000)- 100.csv')
data_test4 = pd.read_csv('./test-1000-100.csv')
data_train5 = pd.read_csv('./train-100(1000)-100.csv')
data_test5 = pd.read_csv('./test-1000-100.csv')
data_train6 = pd.read_csv('./train-150(1000)-100.csv')
data_test6 = pd.read_csv('./test-1000-100.csv')
def Q1_a(data_train,data_test):
    y_train = data_train.loc[:,'y']
    x_train = data_train.loc[:,list(data_train.columns)[0:-1]]
    x_train.insert(0,'x0',1)
    
    y_test = data_test.loc[:,'y']
    x_test = data_test.loc[:,list(data_test.columns)[0:-1]]
    x_test.insert(0,'x0',1)
    
    
    def mse(lamda,x_train,y_train,x_test,y_test):
        w = np.linalg.inv(np.dot(x_train.T,x_train)+lamda*np.eye(x_train.shape[1])).dot(x_train.T).dot(y_train)
        m_train = 1/len(x_train)*sum((x_train.dot(w)-y_train)**2)
        m_test = 1/len(x_test)*sum((x_test.dot(w)-y_test)**2)
        return (m_train,m_test)
    mse_train = []
    mse_test = []
    for lamda in range(0,151):  
        mse_train.append(mse(lamda,x_train,y_train,x_test,y_test)[0])
        mse_test.append(mse(lamda,x_train,y_train,x_test,y_test)[1])
    x = np.arange(0,151,1)    
    plt.plot(x,mse_train,'bo',x,mse_test,'ro')
    min_index_test = mse_test.index(min(mse_test))
    min_lamda = x[min_index_test]
    return min_lamda
plt.subplot(321)
print(Q1_a(data_train1,data_test1))
plt.subplot(322)
print(Q1_a(data_train2,data_test2))
plt.subplot(323)
print(Q1_a(data_train3,data_test3))
plt.subplot(324)
print(Q1_a(data_train4,data_test4))
plt.subplot(325)
print(Q1_a(data_train5,data_test5))
plt.subplot(326)
print(Q1_a(data_train6,data_test6))
plt.show()
def Q1_b(data_train,data_test):
    y_train = data_train.loc[:,'y']
    x_train = data_train.loc[:,list(data_train.columns)[0:-1]]
    x_train.insert(0,'x0',1)
    
    y_test = data_test.loc[:,'y']
    x_test = data_test.loc[:,list(data_test.columns)[0:-1]]
    x_test.insert(0,'x0',1)
    
    
    def mse(lamda):
        w = np.linalg.inv(np.dot(x_train.T,x_train)+lamda*np.eye(x_train.shape[1])).dot(x_train.T).dot(y_train)
        m_train = 1/len(x_train)*sum((x_train.dot(w)-y_train)**2)
        m_test = 1/len(x_test)*sum((x_test.dot(w)-y_test)**2)
        return (m_train,m_test)
    mse_train = []
    mse_test = []
    for lamda in range(1,151):  
        mse_train.append(mse(lamda)[0])
        mse_test.append(mse(lamda)[1])
    x = np.arange(1,151,1)    
    plt.plot(x,mse_train,'bo',x,mse_test,'ro')
    return
plt.subplot(221)
Q1_b(data_train2,data_test2)
plt.subplot(222)
Q1_b(data_train4,data_test4)
plt.subplot(223)
Q1_b(data_train5,data_test5)
plt.show()