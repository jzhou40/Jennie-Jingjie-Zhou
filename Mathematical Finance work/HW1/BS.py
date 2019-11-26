__author__ = 'student'
from math import log
from math import exp
from math import sqrt
from scipy.stats import norm
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

def secantsolve(target,targetfunction,start=None,bounds=None,tols=[0.01,0.01],maxiter=100):

    if (bounds==None and start==None):
        return "NaN"
    elif bounds==None:
        n=0
        a=np.array(start)
        while ((targetfunction(start+tols[0]*2**n)-target)*(targetfunction(start-tols[0]*2**n)-target)>0):
            a=np.append(a,start-tols[0]*2**n)
            a=np.append(a,start+tols[0]*2**n)
            n+=1
            if n==maxiter:
                return "NaN"
        bounds=[a[-2]-tols[0]*2**(n-1),a[-1]+tols[0]*2**(n-1)]
        a=np.append(a,bounds[0])
        a=np.append(a,bounds[1])
        a=np.append(a,a[-2])
        a=np.append(a,a[-4])
        a=np.append(a,start)
        x1=bounds[0]
        x2=start
    elif start==None:
        if ((targetfunction(bounds[0])-target)*(targetfunction(bounds[1])-target)>0):
             return "NaN"
        x2=(bounds[0]+bounds[1])*0.5
        x1=bounds[0]
        a=np.array([bounds[0],bounds[1],x2])
    else:
        if ((targetfunction(bounds[0])-target)*(targetfunction(bounds[1])-target)>0):
             return "NaN"
        x1=bounds[0]
        x2=start
        a=np.array([bounds[0],bounds[1],x2])
    i=0
    while (abs((x2)-(x1))>=tols[0] and abs((targetfunction(x2))-(target))>=tols[1]) :
        temp=x2
        x2=x2+(target-targetfunction(x2))*(x1-x2)/(targetfunction(x1)-targetfunction(x2))
        x1=temp
        a=np.append(a,x2)
        if (bounds==None):
            if (a[i]>bounds[0]):
                return "NaN"
        else:
            if (a[i]>bounds[1]):
                return "NaN"
        i+=1
        if (i==maxiter):
            return "NaN"
    return a

def newton(target,targetfunction,start=None,bounds=None,tols=[0.01,0.01],maxiter=100):
    if (bounds==None and start==None):
        return "NaN"
    elif bounds==None:
        n=0
        a=np.array(start)
        while ((targetfunction(start+tols[0]*2**n)-target)*(targetfunction(start-tols[0]*2**n)-target)>0):
            a=np.append(a,start-tols[0]*2**n)
            a=np.append(a,start+tols[0]*2**n)
            n+=1
            if n==maxiter:
                return "NaN"
        bounds=[a[-2]-tols[0]*2**(n-1),a[-1]+tols[0]*2**(n-1)]
        a=np.append(a,bounds[0])
        a=np.append(a,bounds[1])
        a=np.append(a,a[-2])
        a=np.append(a,a[-4])
        a=np.append(a,start)
        x1=bounds[0]
        x2=start
    elif start==None:
        if ((targetfunction(bounds[0])-target)*(targetfunction(bounds[1])-target)>0):
             return "NaN"
        x2=(bounds[0]+bounds[1])*0.5
        x1=bounds[0]
        a=np.array([bounds[0],bounds[1],x2])
    else:
        if ((targetfunction(bounds[0])-target)*(targetfunction(bounds[1])-target)>0):
             return "NaN"
        x1=bounds[0]
        x2=start
        a=np.array([bounds[0],bounds[1],x2])
    def d(targetfunction):
        def calc():
            dx=0.0000001
            return (targetfunction(x2+dx)-targetfunction(x2))/dx
        return calc()
    i=0
    while (abs((targetfunction(x2))-(target))>tols[1] ):
        x2=x2+(target-targetfunction(x2))/d(targetfunction)
        a=np.append(a,x2)
        if (bounds==None):
            if (a[i]>bounds[0]):
                return "NaN"
        else:
            if (a[i]>bounds[1]):
                return "NaN"
        i+=1
        if (i==maxiter):
            return "NaN"

    return a
def bs(callput,s0,k,r,T,sigma,q=0):
    d1=1/(sigma*sqrt(T))*(log(s0/k)+(r-q+sigma**2/2)*(T))
    d2=d1-sigma*sqrt(T)
    call=norm.cdf(d1)*s0*exp(-q*T)-norm.cdf(d2)*k*exp(-r*T)
    put=k*exp(-r*(T))*norm.cdf(-d2)-s0*norm.cdf(-d1)*exp(-q*(T))
    vega=k*exp(-r*T)*norm.pdf(d2)*sqrt(T)
    if (callput=="Call"):
        delta=exp(-q*T)*norm.cdf(d1)
        return (call,delta,vega)
    elif (callput=="Put"):
        delta=-exp(-q*T)*norm.cdf(-d1)
        return (put,delta,vega)


def bsimpvolsec(callput,S0,K,r,T,price,q=0,priceTolerance=0.01,reportCalls=False):
    def f(sigma):
        d1=1/(sigma*sqrt(T))*(log(S0/K)+(r-q+sigma**2/2)*(T))
        d2=1/(sigma*sqrt(T))*(log(S0/K)+(r-q-sigma**2/2)*(T))
        call=norm.cdf(d1)*S0*exp(-q*T)-norm.cdf(d2)*K*exp(-r*T)
        put=K*exp(-r*(T))*norm.cdf(-d2)-S0*norm.cdf(-d1)*exp(-q*(T))
        if (callput=="Call"):
            return call
        elif (callput=="Put"):
            return put
    if (price<(S0-K))or(S0=="NaN"):
        print "NaN"
    else:
        return secantsolve(price,f,0.5,[-0.5,2.0],[0.0,priceTolerance])

def bsimpvolnewton(callput,S0,K,r,T,price,q=0,priceTolerance=0.01,reportCalls=False):
    def f(sigma):
        d1=1/(sigma*sqrt(T))*(log(S0/K)+(r-q+sigma**2/2)*(T))
        d2=1/(sigma*sqrt(T))*(log(S0/K)+(r-q-sigma**2/2)*(T))
        call=norm.cdf(d1)*S0*exp(-q*T)-norm.cdf(d2)*K*exp(-r*T)
        put=K*exp(-r*(T))*norm.cdf(-d2)-S0*norm.cdf(-d1)*exp(-q*(T))
        if (callput=="Call"):
            return call
        elif (callput=="Put"):
            return put
    return newton(price,f,0.5,[-0.5,2.0],[0.0,priceTolerance])

binaryReadMode='rb'
fileObject = open('TSLAOptions.csv',binaryReadMode)
optsTableR = csv.reader(fileObject)
optsTable = [row for row in optsTableR]
for c in range(81,100):
    a=optsTable[c]
    price=float(a[4])
    S0=float(a[7])
    r=float(a[8])/100
    K=float(a[9])
    callput=a[10]
    q=float(a[6])
    T0=datetime.strptime(a[11],"%m/%d/%Y")
    t=datetime.strptime(a[1],"%m/%d/%Y")
    T1= (relativedelta(T0,t).days)/365.25
    T2=(relativedelta(T0,t).months)/12
    T3=(relativedelta(T0,t).years)
    T=T3+T2+T1
    b=bsimpvolsec(callput,S0,K,r,T,price)
    d=bsimpvolnewton(callput,S0,K,r,T,price)
    print (d[-1]*100,len(bsimpvolnewton(callput,S0,K,r,T,price)))
    #print (callput,S0,K,r,T,price)
