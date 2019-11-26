__author__ = 'student'
import numpy as np
calls=0
def secantsolve(target,targetfunction,start=None,bounds=None,tols=[0.01,0.01],maxiter=100):

    if (bounds==None and start==None):
        raise Exception("bad input")
    elif bounds==None:
        n=0
        a=np.array(start)
        while ((targetfunction(start+tols[0]*2**n)-target)*(targetfunction(start-tols[0]*2**n)-target)>0):
            a=np.append(a,start-tols[0]*2**n)
            a=np.append(a,start+tols[0]*2**n)
            n+=1
            if n==maxiter:
                print ("can not find root")
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
             raise Exception("bad input")
        x2=(bounds[0]+bounds[1])*0.5
        x1=bounds[0]
        a=np.array([bounds[0],bounds[1],x2])
    else:
        if ((targetfunction(bounds[0])-target)*(targetfunction(bounds[1])-target)>0):
             raise Exception("bad input")
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
                raise Exception("bad input")
        else:
            if (a[i]>bounds[1]):
                raise Exception("bad input")
        i+=1
        if (i==maxiter):
            raise Exception("bad input")

    return a

#def secantTestTarget(x):
 #   return (x-0.2)**3-(x-0.2)-0.15
#print secantsolve(0.05, secantTestTarget, 0.5, None, [3e-7, 1e-4])

def newton(target,targetfunction,start=None,bounds=None,tols=[0.01,0.01],maxiter=100):
    if (bounds==None and start==None):
        raise Exception("bad input")
    elif bounds==None:
        n=0
        a=np.array(start)
        while ((targetfunction(start+tols[0]*2**n)-target)*(targetfunction(start-tols[0]*2**n)-target)>0):
            a=np.append(a,start-tols[0]*2**n)
            a=np.append(a,start+tols[0]*2**n)
            n+=1
            if n==maxiter:
                print ("can not find root")
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
             raise Exception("bad input")
        x2=(bounds[0]+bounds[1])*0.5
        x1=bounds[0]
        a=np.array([bounds[0],bounds[1],x2])
    else:
        if ((targetfunction(bounds[0])-target)*(targetfunction(bounds[1])-target)>0):
             raise Exception("bad input")
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
                raise Exception("bad input")
        else:
            if (a[i]>bounds[1]):
                raise Exception("bad input")
        i+=1
        if (i==maxiter):
            raise Exception("bad input")

    return a
#print newton(0.05, secantTestTarget, 1.1, [1.0,1.5], [1e-6,0.01])