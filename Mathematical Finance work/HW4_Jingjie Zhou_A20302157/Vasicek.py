__author__ = 'student'
import numpy as np
from scipy.special import ndtri
from numpy import exp
from numpy import array
from numpy import r_,ones,shape,cast,float64
import math
def VasicekLimits(r0, sigma, kappa, theta,T,prob=1e-6):
    mean=theta*(1-exp(-kappa*T))+r0*exp(-kappa*T)
    if kappa==0:
        var=sigma**2*T
    else:
        var=sigma**2/2/kappa*(1-exp(-2*kappa*T))
    max=mean+var**0.5*ndtri(1-prob)
    min=mean+var**0.5*ndtri(prob)
    return (min,max)
#print VasicekLimits(0.04, 0.02, 0., 0.07,5,1e-2)

def VasicekParams(r0, M,sigma, kappa, theta,T,prob=1e-6):
    dtau=T*M**(-1)
    dr1=2*sigma*dtau**0.5
    N=int(math.ceil((VasicekLimits(r0, sigma, kappa, theta,T,prob)[1]-VasicekLimits(r0, sigma, kappa, theta,T,prob)[0])/dr1))
    min=VasicekLimits(r0, sigma, kappa, theta,T,prob)[0]
    dr=(VasicekLimits(r0, sigma, kappa, theta,T,prob)[1]-VasicekLimits(r0, sigma, kappa, theta,T,prob)[0])/(N-1)
    return (min,dr,N,dtau)
#print VasicekParams(0.04,250, 0.03, 0.5, 0.07,5,1e-3)

def VasicekDiagonals(sigma, kappa, theta,rMin, dr, N,dtau):
    sub=np.array([0])
    dia=np.array([1+rMin*dtau+kappa*(theta-rMin)*dtau/dr])
    super=np.array([-kappa*(theta-rMin)*dtau/dr])
    for i in range(1,N-1):
        sub=np.append(sub,(theta-(rMin+i*dr))*dtau*kappa/dr/2-dtau*sigma**2/dr**2/2)
        dia=np.append(dia,1+(rMin+i*dr)*dtau+dtau*sigma**2/dr**2)
        super=np.append(super,-kappa*(theta-(rMin+i*dr))/2*dtau/dr-sigma**2/2*dtau/dr**2)
    s=(theta-rMin-(N-1)*dr)*dtau*kappa/dr
    sub=np.append(sub,s)
    d=1+(rMin+(N-1)*dr)*dtau-kappa*(theta-(rMin+(N-1)*dr))*dtau/dr
    dia=np.append(dia,d)
    super=np.append(super,0)
    return (sub,dia,super)
#print VasicekDiagonals(0.033, 0.4554, 0.06882,-0.0211, 0.012, 13,0.02)

def CheckExercise(V,eex):
    S=np.ones(np.shape(V),bool)
    S[eex>V]=False
    return S
#print CheckExercise(array([1,0.01,100,0.5]),array([0.75,0.011, 99, 0.51]))

def CallExercise(R, ratio, tau):
    k=ratio*np.exp(-R*tau)
    return k
#print CallExercise(0.02, 1.0, 1)

def TridiagonalSolve( subd,  d,  superd, old ):
    subd, d, superd, old = map(array, (subd, d, superd, old))
    for it in xrange(1, len(d)):
        mc = subd[it]/d[it-1]
        d[it] = d[it] - mc*superd[it-1]
        old[it] = old[it] - mc*old[it-1]

    new = subd
    new[-1] = old[-1]/d[-1]

    for il in xrange(len(d)-2, -1, -1):
        new[il] = (old[il]-superd[il]*new[il+1])/d[il]
    return new
#print TridiagonalSolve( array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ]), array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014]),  array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ]), array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.99561612,
                            #0.99342926,  0.99124749,  0.98907061,  0.98689808,  0.98472858]) )

def VasicekPolicyDiagonals(subdiagonal,diagonal, superdiagonal,vOld, vNew, eex):
    subdiagonal[vNew>eex]=0
    superdiagonal[vNew>eex]=0
    for i in range(1,len(vOld)+1):
        if vNew[i-1]>eex[i-1]:
            diagonal[i-1]=vOld[i-1]*eex[i-1]**(-1)
    return (subdiagonal,diagonal,superdiagonal)
#print VasicekPolicyDiagonals(array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ]),array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014]), array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ]),r_[1:11], 5*r_[1:11], 35.1 *ones(shape(5*r_[1:11])))

def Iterate(subdiagonal, diagonal,superdiagonal,vOld, eex,maxPolicyIterations=10):
    c=[0]
    b=TridiagonalSolve(subdiagonal,diagonal,superdiagonal,vOld)
    if maxPolicyIterations==0:
        i=1
        for a in range(0,len(vOld)):
            if b[a]>eex[a]:
                b[a]=eex[a]
    else:
        i=0
        while np.array_equal(b,c)==False and i<=maxPolicyIterations:
            if i%2==0:
                a=VasicekPolicyDiagonals(subdiagonal,diagonal, superdiagonal,vOld, b, eex)
                c=TridiagonalSolve(a[0],a[1],a[2],vOld)
            elif i%2==1:
                a=VasicekPolicyDiagonals(subdiagonal,diagonal, superdiagonal,vOld, c, eex)
                b=TridiagonalSolve(a[0],a[1],a[2],vOld)
            i=i+1

    return (b,i-1)
#print Iterate(array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ]), array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014]),array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ]),array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.94561612,
 #                           0.94342926,  0.94124749,  0.93907061,  0.93689808,  0.93472858]), 0.942* ones(shape(array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.94561612,
  #                          0.94342926,  0.94124749,  0.93907061,  0.93689808,  0.93472858]))),maxPolicyIterations=0)

def VasicekCallableZCBVals(r0, R, ratio, T,sigma, kappa, theta, M,prob=1e-6,maxPolicyIterations=10):
    (rMin,dr,N,dtau)=VasicekParams(r0, M,sigma, kappa, theta,T,prob)
    r=np.array([])
    vOld=np.array([])
    for i in range(0,N):
        r=np.append(r,rMin+i*dr)
    eex=np.array([])
    vOld=CallExercise(R, ratio, 0)*ones(N)
    vOld[vOld>1]=1
    for i in range(0,M):
        eex=CallExercise(R, ratio, (i+1)*dtau)*ones(N)
        subdiagonal=VasicekDiagonals(sigma, kappa, theta,rMin, dr, N,dtau)[0]
        diagonal=VasicekDiagonals(sigma, kappa, theta,rMin, dr, N,dtau)[1]
        superdiagonal=VasicekDiagonals(sigma, kappa, theta,rMin, dr, N,dtau)[2]
        vOld=Iterate(subdiagonal, diagonal,superdiagonal,vOld, eex,maxPolicyIterations)[0]
    return (r,vOld)
#print VasicekCallableZCBVals(0.04, 0.02, 1.0, 5,0.03, 0.5, 0.07, 250,1e-6,maxPolicyIterations=10)

def VasicekCallableZCB(r0, R, ratio, T,sigma, kappa, theta, M,prob=1e-6,maxPolicyIterations=10):
    r=VasicekCallableZCBVals(r0, R, ratio, T,sigma, kappa, theta, M,prob,maxPolicyIterations)[0]
    v=VasicekCallableZCBVals(r0, R, ratio, T,sigma, kappa, theta, M,prob,maxPolicyIterations)[1]
    for i in range (0,len(r)):
        if r0<r[i]:
            break
    v0=v[i-1]+(r0-r[i-1])/(r[i]-r[i-1])*(v[i]-v[i-1])
    return v0

