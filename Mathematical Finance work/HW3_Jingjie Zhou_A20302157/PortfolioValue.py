import numpy as np
import scipy.stats as st
norminv = st.distributions.norm.ppf
norm = st.distributions.norm.cdf
import numpy.linalg
import scipy.stats
def portfolioCorrelationMatrix(rhoDefault,corrEq, corrVol, corrHzd,rhoSame, rhoUnrelated,):
    R=np.eye(len(corrEq)) * (rhoSame-rhoUnrelated) + rhoUnrelated * np.ones((len(corrEq),len(corrEq)))
    D=np.eye(len(corrEq)) * (1-rhoDefault) + rhoDefault * np.ones((len(corrEq),len(corrEq)))
    a=np.concatenate((D,R,R,R),axis=1)
    b=np.concatenate((R,corrEq,R,R),axis=1)
    c=np.concatenate((R,R,corrVol,R),axis=1)
    d=np.concatenate((R,R,R,corrHzd),axis=1)
    corr=np.concatenate((a,b,c,d),axis=0)
    return corr

def cointegratedUniform(rhoDefault,corrEq, corrVol, corrHzd,rhoSame, rhoUnrelated,samples):
    w = norminv( samples )
    corr=portfolioCorrelationMatrix(rhoDefault,corrEq, corrVol, corrHzd,rhoSame, rhoUnrelated,)
    ch = np.linalg.cholesky( corr )
    z = norm(np.dot( w,ch.T ))
    return z

def defaultTimes(hazardRates,samples):
    tau = -np.log(1-samples)/hazardRates
    return tau

def equityPrices(muE, T, S0, sigmaE, defTimes,samples):
    w = norminv( samples )
    S=S0*np.exp((muE-0.5*sigmaE**2)*T+sigmaE*T**0.5*w)
    S[defTimes<T]=0
    return S

def hazardRates(muH, T, h0, sigmaH,samples):
    w = norminv( samples )
    h=h0*np.exp((muH-0.5*sigmaH**2)*T+sigmaH*T**0.5*w)
    return h

def ZCBValues(bondTenors, bondRiskFreeRates, defTimes, h, recoveryRates, T):
    a=np.exp(-h*(bondTenors-T))*np.exp(-bondRiskFreeRates*(bondTenors-T))
    for i in range(0,h.shape[0]):
        for j in range(0,h.shape[1]):
            if defTimes[i][j]<T:
                a[i][j]=recoveryRates[j]
    return a

def volatilities(eta,T, sigma0, varvol,samples):
    w = norminv( samples )
    v=sigma0*np.exp((eta-0.5*varvol**2)*T+varvol*T**0.5*w)
    return v

def optionPrices(callput, ST, K, r, optionTenors,sigmaT, q, defTimes,T):
    ST[defTimes<T]=0
    d1=1/(sigmaT*np.sqrt(optionTenors-T))*(np.log(ST/K)+(r-q+sigmaT**2/2)*(optionTenors-T))
    d2=d1-sigmaT*np.sqrt(optionTenors-T)
    oP=norm(callput*d1)*ST*np.exp(-q*(optionTenors-T))*callput-norm(callput*d2)*K*np.exp(-r*(optionTenors-T))*callput
    return oP

def portfolioValueElements(T,eqPositions, bondPositions, optionPositions,S0, h0, sigma0,muE, muH, eta,sigmaH, varvol,bondTenors, bondRiskFreeRates, recoveryRates,callput, K, r, optionTenors, q,rhoDefault,corrEq, corrVol, corrHzd,rhoSame, rhoUnrelated,samples):
    cU=cointegratedUniform(rhoDefault,corrEq, corrVol, corrHzd,rhoSame, rhoUnrelated,samples)
    samplesd=np.split(cU,4,axis=1)[0]
    samplese=np.split(cU,4,axis=1)[1]
    samplesv=np.split(cU,4,axis=1)[2]
    samplesh=np.split(cU,4,axis=1)[3]
    h=hazardRates(muH, T, h0, sigmaH,samplesh)
    vol=volatilities(eta,T, sigma0, varvol,samplesv)
    defTimes=defaultTimes(h,samplesd)
    sigmaE=sigma0
    eP=equityPrices(muE, T, S0, sigmaE, defTimes,samplese)

    ST=eP
    sigmaT=vol
    oP=optionPrices(callput, ST, K, r, optionTenors,sigmaT, q, defTimes,T)

    bond=ZCBValues(bondTenors, bondRiskFreeRates, defTimes, h, recoveryRates, T)

    opvalue=oP*optionPositions
    eqvalue=eqPositions*eP
    bondvalue=bond*bondPositions
    value=np.concatenate((eqvalue,bondvalue,opvalue),axis=1)
    return value

def portfolioValue(T,eqPositions, bondPositions, optionPositions,S0, h0, sigma0,muE, muH, eta,sigmaH, varvol,bondTenors, bondRiskFreeRates, recoveryRates,callput, K, r, optionTenors, q,rhoDefault,corrEq, corrVol, corrHzd,rhoSame, rhoUnrelated,samples):
    value=portfolioValueElements(T,eqPositions, bondPositions, optionPositions,S0, h0, sigma0,muE, muH, eta,sigmaH, varvol,bondTenors, bondRiskFreeRates, recoveryRates,callput, K, r, optionTenors, q,rhoDefault,corrEq, corrVol, corrHzd,rhoSame, rhoUnrelated,samples)
    sum=np.zeros(len(value))
    for i in range (0,len(value)):
        sum[i]=np.sum(value[i][:])
    return sum

def expectedShortfall(valuations, level=0.05, weights=1):
    sortv=np.argsort(valuations)
    M=len(valuations)
    weights=weights*np.ones(M)
    sum=0
    k=0
    value=0
    w=0
    while sum<=level:
        k=k+1
        sum+=weights[sortv[k]]/M
    for i in range (0,k-1):
        value+=valuations[sortv[i]]*weights[sortv[i]]
        w+=weights[sortv[i]]
    a=value/w
    return a

def importanceSampleDetails(originalUniformSamples, correlation_in,shifts_in=0):
    if type(correlation_in)==float:
        M=len(originalUniformSamples[0])
        correlation_in=correlation_in*np.ones((M,M))+(1-correlation_in)*np.eye(M)
    ch = numpy.linalg.cholesky(correlation_in)


    samples=scipy.special.ndtri(originalUniformSamples)
    samples=np.dot(samples,np.transpose(ch))+shifts_in
    weights=np.exp(-np.dot(shifts_in,np.linalg.solve(correlation_in,np.transpose(samples)))+0.5*np.dot(shifts_in,np.linalg.solve(correlation_in,np.transpose(shifts_in))))
    samples=st.distributions.norm.cdf(samples)
    dict1={'Weights':weights,'Samples':samples}

    return dict1


