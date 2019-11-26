from __future__ import division
__author__ = 'student'
from math import exp
from numpy import array,sqrt,ndarray
from numpy import append
import numpy as np
from numpy import exp,sqrt,array
from numpy import append,maximum
import scipy.special as sc
import copy
import types
from numpy.random import rand
from numpy.random import randn,rand
from BSPaths import MakeBSPaths
def MCRangePayoff(path, r, t,fixingTimes, finalT, fixingValues,KLow, KHigh, coupon):
    i=0
    j=0
    value=0
    fixingTimes1=append(fixingTimes,finalT)
    while i<(len(fixingTimes1)-1):
        if KLow<path[i]<KHigh:
            optionvalue=exp(-r*(fixingTimes1[i+1]-t))*coupon*(fixingTimes1[i+1]-fixingTimes1[i])
            value+=optionvalue
        i=i+1
    return value

def MCRangeValues(S0, KLow, KHigh, coupon, sigma, r, t, q,fixingTimes, finalT, fixingValues, samples=1, shift=0, integrationType='strong',):
    paths=MakeBSPaths(S0, sigma, r, t, q,fixingTimes, fixingValues,samples,shift,integrationType)['Paths']
    weights=MakeBSPaths(S0, sigma, r, t, q,fixingTimes, fixingValues,samples,shift,integrationType)['Weights']
    a=[]
    if type(samples) is types.IntType:
        M=samples
    else:
        M=len(samples)
    for i in range (0,M):
        path=paths[i,]
        v=MCRangePayoff(path, r, t,fixingTimes, finalT, fixingValues,KLow, KHigh, coupon)
        a=append(a,v)
    dict2={}
    dict2.setdefault('Samples',a)
    dict2.setdefault('Weights',weights)
    return dict2

def MCRange(S0,KLow,KHigh,coupon,sigma,r,t,q,fixingTimes,finalT,fixingValues,samples,shift=0,integrationType='strong',):
    weight=MCRangeValues(S0, KLow, KHigh, coupon, sigma, r, t, q,fixingTimes, finalT, fixingValues, samples, shift, integrationType,)['Weights']
    sample=MCRangeValues(S0, KLow, KHigh, coupon, sigma, r, t, q,fixingTimes, finalT, fixingValues, samples, shift, integrationType,)['Samples']
    c=np.sum(weight*sample)/np.sum(weight)
    dict3={}
    dict3.setdefault('Std Err',np.std(samples)/np.sqrt(len(samples)))
    dict3.setdefault('Mean',c)
    return dict3

