from __future__ import division
__author__ = 'student'
import numpy as np
from numpy import exp,sqrt,array
from numpy import append,maximum
import scipy.special as sc
import copy
import types
from numpy.random import randn,rand
def MakeBSPaths(S0, sigma, r, t, q,fixingTimes, fixingValues,samples,shift=0,integrationType='strong'):
    M=len(samples)
    impSampZ=sc.ndtri(samples)
    N=len(fixingTimes)
    fixingTimes1=np.append(t,fixingTimes)
    dict={}
    ST=S0
    b=copy.deepcopy(impSampZ)
    c=copy.deepcopy(impSampZ)
    e=copy.deepcopy(impSampZ)
    for j in range(0,M,1):
        for i in range(1,len(fixingTimes1),1):
            shifta=shift*(fixingTimes1[i]-maximum(t,fixingTimes1[i-1]))/(fixingTimes1[-1]-t)
            c[j,i-1]=impSampZ[j,i-1]+shifta
            e[j,i-1]=exp(-shifta*(impSampZ[j,i-1]+shifta)+(shifta)**2/2.0)
            if integrationType=='strong':
                ST*=exp((r-q-0.5*sigma**2)*(fixingTimes1[i]-fixingTimes1[i-1])+sigma*sqrt(fixingTimes1[i]-fixingTimes1[i-1])*c[j,i-1])
                b[j,i-1]=ST
            elif integrationType=='euler':
                ST*=1+(r-q)*(fixingTimes1[i]-fixingTimes1[i-1])+sigma*sqrt(fixingTimes1[i]-fixingTimes1[i-1])*c[j,i-1]
                b[j,i-1]=ST
            elif integrationType=='milstein':
                ST*=1+(r-q)*(fixingTimes1[i]-fixingTimes1[i-1])++sigma*sqrt(fixingTimes1[i]-fixingTimes1[i-1])*c[j,i-1]+0.5*sigma**2*(c[j,i-1]**2*(fixingTimes1[i]-fixingTimes1[i-1])-(fixingTimes1[i]-fixingTimes1[i-1]))
                b[j,i-1]=ST
        ST=S0
    weights=np.prod(e,axis=1)
    dict.setdefault('Paths',b)
    dict.setdefault('Weights',weights)
    return dict



