#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:16:24 2019

@author: chengzhao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_train1 = pd.read_csv('./train-1000-100.csv')
data_test1 = pd.read_csv('./test-1000-100.csv')
data_train1.insert(0,'x0',1)
data_test1.insert(0,'x0',1)
def mse(lamda,x_train,y_train,x_test,y_test):
        w = (np.linalg.inv(np.dot(x_train.T,x_train)+lamda*np.eye(x_train.shape[1]))).dot(x_train.T).dot(y_train)
        m_test = 1/len(x_test)*sum((x_test.dot(w)-y_test)**2)
        return m_test

def lc(lamda,data_train,data_test,color):
    y_axis = [] 
    x_axis = []
    for i in range(10,len(data_train)+1,10):
        t_mse = 0
        for t in range(10):
            s_train = data_train.sample(n=i)
            t_mse += mse(lamda,s_train.loc[:,list(s_train.columns)[0:-1]],s_train.loc[:,'y'],data_test.loc[:,list(data_test.columns)[0:-1]],data_test.loc[:,'y'])
        a_mse = t_mse/10
        y_axis.append(a_mse)
        x_axis.append(i)
    plt.plot(x_axis,y_axis,color)
plt.subplot(221)
lc(1,data_train1,data_test1,'r')
plt.subplot(222)
lc(25,data_train1,data_test1,'g')
plt.subplot(223)
lc(150,data_train1,data_test1,'b')