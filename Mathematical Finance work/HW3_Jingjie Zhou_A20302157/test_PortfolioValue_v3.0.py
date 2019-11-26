from __future__ import division
import unittest
import datetime
from copy import copy
import numpy as np
from PortfolioValue import *
np.set_printoptions(linewidth=480)


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class TestPortfolioValue(unittest.TestCase):
    def confirm( self, result, expected, digits=7, logOnErr=True):
        res = copy( result )
        try:
            self.confirm_iter( res, expected, digits=digits)
        except:
            if logOnErr:
                print('Did Not Match Expected:\n\t\t%r'%result)
            raise
    
    def confirm_iter( self, result, expected, digits=7):
        ''' Nest iterables and check '''
        if hasattr(expected,'__iter__'):
            if type(expected)==dict:
                for (key,expectedElem) in expected.iteritems():
                    resElem = result[key]
                    self.confirm_iter(resElem, expectedElem, digits )
            else:
                for (i, expectedElem) in enumerate(expected):
                    resElem = result[i]
                    self.confirm_iter(resElem, expectedElem, digits )
        else:
            self.assertAlmostEqual(result, expected, digits)
    
    def test01CorrelMatrix(self):
        'Correlation Matrix'
        ce = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        cv = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        ch = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])

        i_got = portfolioCorrelationMatrix(0.45,ce,cv,ch,0.1,0.05)
        i_exp = np.array([[ 1.   ,  0.45 ,  0.45 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ],
                       [ 0.45 ,  1.   ,  0.45 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ],
                       [ 0.45 ,  0.45 ,  1.   ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ],
                       [ 0.1  ,  0.05 ,  0.05 ,  1.   ,  0.11 ,  0.22 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ],
                       [ 0.05 ,  0.1  ,  0.05 ,  0.11 ,  1.   ,  0.33 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ],
                       [ 0.05 ,  0.05 ,  0.1  ,  0.22 ,  0.33 ,  1.   ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ],
                       [ 0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  1.   ,  0.101,  0.202,  0.1  ,  0.05 ,  0.05 ],
                       [ 0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.101,  1.   ,  0.303,  0.05 ,  0.1  ,  0.05 ],
                       [ 0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.202,  0.303,  1.   ,  0.05 ,  0.05 ,  0.1  ],
                       [ 0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  1.   ,  0.01 ,  0.02 ],
                       [ 0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.01 ,  1.   ,  0.03 ],
                       [ 0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.05 ,  0.05 ,  0.1  ,  0.02 ,  0.03 ,  1.   ]])
        self.confirm( i_got, i_exp )
            


    def test10CointegratedUniform(self):
        'Correlation Matrix'
        ce = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        cv = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        ch = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])

        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((17,12)),axis=0), axis=1) ))
        i_got = cointegratedUniform(0.45,ce,cv,ch,0.1,0.05,samples)
        i_exp = np.array([[ 0.84147098,  0.94988244,  0.46325575,  0.78690124,  0.97468851,  0.55770313,  0.73198635,  0.99590565,  0.71666112,  0.66539553,  0.99999893,  0.61515772],
                       [ 0.90929743,  0.88933413,  0.61716315,  0.99237706,  0.68300516,  0.74396861,  0.99626561,  0.45855558,  0.86012815,  0.96944104,  0.01965442,  0.93811221],
                       [ 0.14112001,  0.1572244 ,  0.20201461,  0.49187645,  0.61265659,  0.74941111,  0.813747  ,  0.90683545,  0.98322599,  0.98906685,  0.99992013,  0.99540625],
                       [ 0.7568025 ,  0.99110103,  0.84835543,  0.31781052,  0.93586791,  0.94727026,  0.30733783,  0.65998345,  0.9916248 ,  0.77062726,  0.04183874,  0.86025445],
                       [ 0.95892427,  0.81083928,  0.87299579,  0.93678007,  0.19415085,  0.98448953,  0.54579088,  0.76351704,  0.922393  ,  0.37747728,  0.99983005,  0.4736642 ],
                       [ 0.2794155 ,  0.42815345,  0.63217535,  0.89475132,  0.99070653,  0.99934343,  0.9392885 ,  0.86504834,  0.83194621,  0.42603114,  0.06235477,  0.4062587 ],
                       [ 0.6569866 ,  0.98870566,  0.95212167,  0.29135253,  0.49399923,  0.89616333,  0.95381524,  0.65591911,  0.37419794,  0.81120584,  0.99977282,  0.81626583],
                       [ 0.98935825,  0.70424359,  0.97707386,  0.64099213,  0.76965416,  0.86643472,  0.63017418,  0.93722926,  0.53005952,  0.99734131,  0.05978204,  0.9914667 ],
                       [ 0.41211849,  0.69328079,  0.9381722 ,  0.9912349 ,  0.90920783,  0.86856544,  0.23866083,  0.31334966,  0.59799248,  0.91649239,  0.99946744,  0.95998764],
                       [ 0.54402111,  0.89679282,  0.99044675,  0.7529205 ,  0.33453675,  0.38514966,  0.78853225,  0.99586131,  0.98609649,  0.59043511,  0.08997986,  0.71819077],
                       [ 0.99999021,  0.42200354,  0.99999567,  0.04803597,  0.99942571,  0.23321772,  0.99984188,  0.10095138,  0.99968684,  0.13832863,  0.99939881,  0.17618467],
                       [ 0.53657292,  0.88777096,  0.99269761,  0.77508556,  0.38214983,  0.35616068,  0.75194078,  0.98843137,  0.98835069,  0.65617667,  0.1003956 ,  0.63997732],
                       [ 0.42016704,  0.7079685 ,  0.9487149 ,  0.98592596,  0.88945261,  0.83022747,  0.15612512,  0.37136415,  0.65175495,  0.94220035,  0.99884149,  0.93046877],
                       [ 0.99060736,  0.69587705,  0.98037622,  0.61445478,  0.7937752 ,  0.84755675,  0.67774703,  0.91452258,  0.60123556,  0.99072208,  0.09790703,  0.9987553 ],
                       [ 0.65028784,  0.98574985,  0.95378742,  0.32543142,  0.45374182,  0.87017576,  0.97051919,  0.70976636,  0.27169183,  0.76348041,  0.99889611,  0.86280683],
                       [ 0.28790332,  0.44577414,  0.65777745,  0.91083561,  0.99551124,  0.99893353,  0.92083209,  0.83446259,  0.76849288,  0.3263052 ,  0.14299299,  0.51340645],
                       [ 0.96139749,  0.80520785,  0.88286453,  0.92539785,  0.24487013,  0.99344484,  0.49020003,  0.80631621,  0.90349486,  0.4690194 ,  0.99806623,  0.35346073]])
        self.confirm( i_got, i_exp )

    def test20DefaultTimes(self):
        """Default times"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((3,4)),axis=0), axis=1) ))
        h = np.array([0.01,0.0001,0.91,0.055])
        i_got = defaultTimes(  h,  samples )
        i_exp = np.array([[  1.84181764e+02,   2.40016955e+04,   1.67171509e-01,
                              2.57069344e+01],
                           [  2.40016955e+02,   1.41388139e+04,   3.60101745e-01,
                              8.25994549e+01],
                           [  1.52126073e+01,   3.27692588e+03,   5.83769074e-01,
                              1.39837496e+01]])
        self.confirm( i_got, i_exp,4 )


    def test30EqPrices(self):
        """Equity prices trivial"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((3,4)),axis=0), axis=1) ))
        muE = np.zeros(4)
        T = 1.
        S0 = np.array([1,10,100,1000.])
        sigmaE = np.ones_like(S0)
        defTimes = np.ones_like(samples)*100
        i_got = equityPrices(muE, T, S0, sigmaE, defTimes,
            samples)
        i_exp = np.array([[  1.64958188e+00,   2.30813787e+01,   2.06945265e+01,
                              1.21659261e+03],
                           [  2.30813787e+00,   1.21659261e+01,   3.38044155e+01,
                              6.06732077e+03],
                           [  2.06945265e-01,   3.38044155e+00,   4.85731731e+01,
                              6.64848223e+02]])
        self.confirm( i_got, i_exp,4 )

    def test32EqPrices(self):
        """Equity prices no defaults"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((3,4)),axis=0), axis=1) ))
        muE = np.array([1.,2.,0.5,0.7]) * 0.02
        T = 1./52.
        S0 = np.array([1,10,100,1000.])
        sigmaE = np.array([40.,31.1,60,100.5])*0.01
        defTimes = np.ones_like(samples)*100
        i_got = equityPrices(muE, T, S0, sigmaE, defTimes,
            samples)
        i_exp = np.array([[  1.05584893e+00,   1.05916109e+01,   9.11430736e+01,   1.09151365e+03],
       [  1.07570748e+00,   1.03030883e+01,   9.49415143e+01,   1.36548733e+03],
       [  9.41010661e-01,   9.74946719e+00,   9.78485835e+01,   1.00335759e+03]])
        self.confirm( i_got, i_exp,4 )

    def test35EqPrices(self):
        """Equity prices with defaults"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((3,4)),axis=0), axis=1) ))
        muE = np.array([1.,2.,0.5,0.7]) * 0.02
        T = 1./52.
        S0 = np.array([1,10,100,1000.])
        sigmaE = np.array([40.,31.1,60,100.5])*0.01
        defTimes = np.ones_like(samples)*100
        defTimes[1:3,1:3] = 0.5/52.0
        i_got = equityPrices(muE, T, S0, sigmaE, defTimes,
            samples)
        i_exp = np.array([[  1.05584893e+00,   1.05916109e+01,   9.11430736e+01,   1.09151365e+03],
       [  1.07570748e+00,   0.00000000e+00,   0.00000000e+00,   1.36548733e+03],
       [  9.41010661e-01,   0.00000000e+00,   0.00000000e+00,   1.00335759e+03]])
        self.confirm( i_got, i_exp,4 )

    def test42HzdRates(self):
        """Hazard Rates"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((5,4)),axis=0), axis=1) ))
        muH = np.array([1.,2.,0.5,0.7]) * 0.02
        T = 1.5/52.
        h0 = np.array([0.01,0.0001,0.91,0.055])
        sigmaH = np.array([10.,11.1,20,20.5])*0.01
        i_got = hazardRates(muH, T, h0, sigmaH,
            samples)
        i_exp = np.array([[  1.01757842e-02,   1.02651675e-04,   8.77107868e-01,   5.63378035e-02],
       [  1.02340061e-02,   1.01419829e-04,   8.91850917e-01,   5.95795719e-02],
       [  9.82327635e-03,   9.90005698e-05,   9.02900010e-01,   5.51649215e-02],
       [  1.01232998e-02,   1.04539174e-04,   9.12578900e-01,   5.39279982e-02],
       [  1.03041010e-02,   1.00306533e-04,   9.21747407e-01,   5.76535593e-02]])
        self.confirm( i_got, i_exp,4 )


    def test52Vols(self):
        """Volatilities"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((3,6)),axis=0), axis=1) ))
        eta = np.array([1.,2.,0.5,0.7,0.5,0.7]) * 0.02
        T = 4./52.
        sigma0 = np.array([40.,31.1,60,100.5,20,35])*0.01
        varvol = np.array([110.,111.1,120,120.5,60,70])*0.01
        i_got = volatilities(eta,T, sigma0, varvol,
            samples)
        i_exp = np.array([[ 0.51889879,  0.44908021,  0.39719704,  1.20062635,  0.26362225,  0.30694569],
       [ 0.57489811,  0.36866015,  0.46766616,  2.05416218,  0.20106743,  0.35001889],
       [ 0.27545003,  0.24845611,  0.52763002,  0.98107998,  0.21050112,  0.3921783 ]])
        self.confirm( i_got, i_exp,5 )


    def test60Bonds(self):
        """Bond Values, trivial"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((5,7)),axis=0), axis=1) ))
        recoveryRates = np.zeros(7)
        bondRiskFreeRates = np.ones(7)*0.01
        bondTenors = np.ones(7)*2.0
        defTimes = np.ones_like(samples)*100.0
        h = np.ones_like(samples)*0.05
        T = 1.0
        i_got = ZCBValues(bondTenors, bondRiskFreeRates, defTimes, h, recoveryRates, T)
        i_exp = 0.94176453 * np.ones_like(samples)
        self.confirm( i_got, i_exp,5 )


    def test62Bonds(self):
        """Bond Values, no defaults"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((5,4)),axis=0), axis=1) ))
        recoveryRates = np.array([0.22,0.1,0.1,0.2])
        bondRiskFreeRates = np.array([0.02,0.022,0.013,0.012])
        bondTenors = np.array([5,30,20,7.5])
        defTimes = np.ones_like(samples)*100
        h = np.array([ [0.01,0.0001,0.0091,0.045],
                        [0.02,0.0002,0.0081,0.055],
                        [0.03,0.0003,0.0071,0.035],
                        [0.04,0.0004,0.0061,0.025],
                        [0.05,0.0005,0.0051,0.015],
                        ])
        T = 1.5/52.
        i_got = ZCBValues(bondTenors, bondRiskFreeRates, defTimes, h, recoveryRates, T)
        i_exp = np.array([[ 0.86145314,  0.51563171,  0.64315952,  0.65321055],
                           [ 0.81967599,  0.51408862,  0.65613328,  0.60618667],
                           [ 0.77992486,  0.51255014,  0.66936874,  0.70388223],
                           [ 0.74210152,  0.51101627,  0.68287119,  0.75848467],
                           [ 0.70611246,  0.50948699,  0.696646  ,  0.8173228 ]])
        self.confirm( i_got, i_exp,5 )


    def test65Bonds(self):
        """Bond Values, with defaults"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((5,4)),axis=0), axis=1) ))
        recoveryRates = np.array([0.22,0.1,0.15,0.2])
        bondRiskFreeRates = np.array([0.02,0.022,0.013,0.012])
        bondTenors = np.array([5,30,20,7.5])
        defTimes = np.ones_like(samples)*100
        defTimes[1:3,1:3] = 1.4/52.
        h = np.array([ [0.01,0.0001,0.0091,0.045],
                        [0.02,0.0002,0.0081,0.055],
                        [0.03,0.0003,0.0071,0.035],
                        [0.04,0.0004,0.0061,0.025],
                        [0.05,0.0005,0.0051,0.015],
                        ])
        T = 1.5/52.
        i_got = ZCBValues(bondTenors, bondRiskFreeRates, defTimes, h, recoveryRates, T)
        i_exp = np.array([[ 0.86145314,  0.51563171,  0.64315952,  0.65321055],
                       [ 0.81967599,  0.1       ,  0.15      ,  0.60618667],
                       [ 0.77992486,  0.1       ,  0.15      ,  0.70388223],
                       [ 0.74210152,  0.51101627,  0.68287119,  0.75848467],
                       [ 0.70611246,  0.50948699,  0.696646  ,  0.8173228 ]])
        self.confirm( i_got, i_exp,5 )


    def test72Options(self):
        """Option prices, no defaults"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,5)),axis=0), axis=1) ))
        callput = np.array([-1,1,-1,1,-1])
        ST = np.array([[  1.07676428e+00,   1.10153861e+01,   9.20413747e+01,
                              1.10660434e+03, 102],
                           [  1.09701620e+00,   1.07153196e+01,   9.58772526e+01,
                              1.38436583e+03, 103],
                           [  9.59651169e-01,   1.01395478e+01,   9.88129738e+01,
                              1.01722947e+03, 101],
                           [  1.09701620e+00,   1.07153196e+01,   9.58772526e+01,
                              1.38436583e+03, 91],
                           [  1.2e+00,   1.011e+01,   9.58772526e+01,
                              1.38436583e+03, 93],
                           [  9.59651169e-01,   1.01395478e+01,   9.5e+01,
                              1.05e+03, 100],
                ])
        K = np.array([  1.07e+00,   1.10e+01,   9.2e+01, 1.10e+03, 95])
        r = np.array([ 0.011,0.012,0.013,0.014,0.015])
        q = np.array([ 0.0011,0.0012,0.0013,0.0014,0.0015])
        optionTenors = np.array([0.9,2.1,0.3,0.7,0.25])
        sigmaT = np.array([[ 0.52856744,  0.46597155,  0.40088045,  1.21624283,  0.26606695,],
                           [ 0.58561019,  0.38252664,  0.47200306,  2.08088055,  0.20293203,],
                           [ 0.2805825 ,  0.25780134,  0.17      ,  0.99384083,  0.2124532 ,],
                           [ 0.12      ,  0.38252664,  0.47200306,  2.08088055,  0.20293203,],
                           [ 0.58561019,  0.38252664,  0.47200306,  1.06      ,  0.20293203,],
                           [ 0.2805825 ,  0.25780134,  0.532523  ,  0.33      ,  0.2124532 ,],
                           ])
        defTimes = np.ones_like(samples)*100
        T = 1.5/52.
        i_got = optionPrices(callput, ST, K, r, optionTenors,
            sigmaT, q, defTimes,T)
        i_exp = np.array([[  2.00164583e-01,   2.98235910e+00,   7.46946130e+00,
                              4.26886029e+02,   2.09840245e+00],
                           [  2.14485619e-01,   2.30208124e+00,   7.23558776e+00,
                              9.01591865e+02,   9.83548039e-01],
                           [  1.63230690e-01,   1.24520096e+00,   9.37789217e-01,
                              2.97247894e+02,   1.53478990e+00],
                           [  3.21947560e-02,   2.30208124e+00,   7.23558776e+00,
                              9.01591865e+02,   5.69664496e+00],
                           [  1.79695483e-01,   1.94857641e+00,   7.23558776e+00,
                              5.76433805e+02,   4.49430819e+00],
                           [  1.63230690e-01,   1.24520096e+00,   8.72819025e+00,
                              9.57658227e+01,   1.79486625e+00]])
        self.confirm( i_got, i_exp,5 )


    def test75Options(self):
        """Option prices, all defaults"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,5)),axis=0), axis=1) ))
        callput = np.array([-1,1,-1,1,-1])
        ST = np.array([[  1.07676428e+00,   1.10153861e+01,   9.20413747e+01,
                              1.10660434e+03, 102],
                           [  1.09701620e+00,   1.07153196e+01,   9.58772526e+01,
                              1.38436583e+03, 103],
                           [  9.59651169e-01,   1.01395478e+01,   9.88129738e+01,
                              1.01722947e+03, 101],
                           [  1.09701620e+00,   1.07153196e+01,   9.58772526e+01,
                              1.38436583e+03, 91],
                           [  1.2e+00,   1.011e+01,   9.58772526e+01,
                              1.38436583e+03, 93],
                           [  9.59651169e-01,   1.01395478e+01,   9.5e+01,
                              1.05e+03, 100],
                ])
        K = np.array([  1.07e+00,   1.10e+01,   9.2e+01, 1.10e+03, 95])
        r = np.array([ 0.011,0.012,0.013,0.014,0.015])
        q = np.array([ 0.0011,0.0012,0.0013,0.0014,0.0015])
        optionTenors = np.array([0.9,2.1,0.3,0.7,0.25])
        sigmaT = np.array([[ 0.52856744,  0.46597155,  0.40088045,  1.21624283,  0.26606695,],
                           [ 0.58561019,  0.38252664,  0.47200306,  2.08088055,  0.20293203,],
                           [ 0.2805825 ,  0.25780134,  0.17      ,  0.99384083,  0.2124532 ,],
                           [ 0.12      ,  0.38252664,  0.47200306,  2.08088055,  0.20293203,],
                           [ 0.58561019,  0.38252664,  0.47200306,  1.06      ,  0.20293203,],
                           [ 0.2805825 ,  0.25780134,  0.532523  ,  0.33      ,  0.2124532 ,],
                           ])
        defTimes = np.ones_like(samples)*0.01
        T = 1.5/52.
        i_got = optionPrices(callput, ST, K, r, optionTenors,
            sigmaT, q, defTimes,T)
        i_exp = np.array([[  1.05979549,   0.        ,  91.67627091,   0.        ,
                             94.68537791],
                           [  1.05979549,   0.        ,  91.67627091,   0.        ,
                             94.68537791],
                           [  1.05979549,   0.        ,  91.67627091,   0.        ,
                             94.68537791],
                           [  1.05979549,   0.        ,  91.67627091,   0.        ,
                             94.68537791],
                           [  1.05979549,   0.        ,  91.67627091,   0.        ,
                             94.68537791],
                           [  1.05979549,   0.        ,  91.67627091,   0.        ,
                             94.68537791]])
        self.confirm( i_got, i_exp,5 )

    def test78Options(self):
        """Option prices, some defaults"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,5)),axis=0), axis=1) ))
        callput = np.array([-1,1,-1,1,-1])
        ST = np.array([[  1.07676428e+00,   1.10153861e+01,   9.20413747e+01,
                              1.10660434e+03, 102],
                           [  1.09701620e+00,   1.07153196e+01,   9.58772526e+01,
                              1.38436583e+03, 103],
                           [  9.59651169e-01,   1.01395478e+01,   9.88129738e+01,
                              1.01722947e+03, 101],
                           [  1.09701620e+00,   1.07153196e+01,   9.58772526e+01,
                              1.38436583e+03, 91],
                           [  1.2e+00,   1.011e+01,   9.58772526e+01,
                              1.38436583e+03, 93],
                           [  9.59651169e-01,   1.01395478e+01,   9.5e+01,
                              1.05e+03, 100],
                ])
        K = np.array([  1.07e+00,   1.10e+01,   9.2e+01, 1.10e+03, 95])
        r = np.array([ 0.011,0.012,0.013,0.014,0.015])
        q = np.array([ 0.0011,0.0012,0.0013,0.0014,0.0015])
        optionTenors = np.array([0.9,2.1,0.3,0.7,0.25])
        sigmaT = np.array([[ 0.52856744,  0.46597155,  0.40088045,  1.21624283,  0.26606695,],
                           [ 0.58561019,  0.38252664,  0.47200306,  2.08088055,  0.20293203,],
                           [ 0.2805825 ,  0.25780134,  0.17      ,  0.99384083,  0.2124532 ,],
                           [ 0.12      ,  0.38252664,  0.47200306,  2.08088055,  0.20293203,],
                           [ 0.58561019,  0.38252664,  0.47200306,  1.06      ,  0.20293203,],
                           [ 0.2805825 ,  0.25780134,  0.532523  ,  0.33      ,  0.2124532 ,],
                           ])
        defTimes = np.ones_like(samples)*100
        defTimes[:2,3:] = 0.01
        T = 1.5/52.
        i_got = optionPrices(callput, ST, K, r, optionTenors,
            sigmaT, q, defTimes,T)
        i_exp = np.array([[  2.00164583e-01,   2.98235910e+00,   7.46946130e+00,
                              0.00000000e+00,   9.46853779e+01],
                           [  2.14485619e-01,   2.30208124e+00,   7.23558776e+00,
                              0.00000000e+00,   9.46853779e+01],
                           [  1.63230690e-01,   1.24520096e+00,   9.37789217e-01,
                              2.97247894e+02,   1.53478990e+00],
                           [  3.21947560e-02,   2.30208124e+00,   7.23558776e+00,
                              9.01591865e+02,   5.69664496e+00],
                           [  1.79695483e-01,   1.94857641e+00,   7.23558776e+00,
                              5.76433805e+02,   4.49430819e+00],
                           [  1.63230690e-01,   1.24520096e+00,   8.72819025e+00,
                              9.57658227e+01,   1.79486625e+00]])
        self.confirm( i_got, i_exp,5 )


    def test80Elements(self):
        """Portfolio valuation elements, equities all one share"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.ones(3)
        bondPositions = np.zeros(3)
        optionPositions = np.zeros(3)
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([[  10.66364773,   72.88762586,  252.11352934,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ],
                           [  12.24756055,   62.69288986,  261.01029168,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ],
                           [   9.94966068,   61.49153001,  261.3120656 ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ],
                           [   9.5737958 ,   69.73656697,  278.6632716 ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ],
                           [  11.34811438,   54.69737156,  289.03407284,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ],
                           [  11.0851403 ,   75.91271392,  310.53768175,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ,    0.        ]])
        self.confirm( i_got, i_exp,5 )

    def test81Elements(self):
        """Portfolio valuation elements, equities"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.array([10,100,1000])
        bondPositions = np.zeros(3)
        optionPositions = np.zeros(3)
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([[  1.06636477e+02,   7.28876259e+03,   2.52113529e+05,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                           [  1.22475605e+02,   6.26928899e+03,   2.61010292e+05,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                           [  9.94966068e+01,   6.14915300e+03,   2.61312066e+05,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                           [  9.57379580e+01,   6.97365670e+03,   2.78663272e+05,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                           [  1.13481144e+02,   5.46973716e+03,   2.89034073e+05,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
                           [  1.10851403e+02,   7.59127139e+03,   3.10537682e+05,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])

        self.confirm( i_got, i_exp,3 )

    def test82Elements(self):
        """Portfolio valuation elements, options all one contract"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.zeros(3)
        bondPositions = np.zeros(3)
        optionPositions = np.ones(3)
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([[  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.26645299,   4.82139526,  28.68956466],
                           [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.12656246,   5.93050282,  35.25340618],
                           [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.44324916,   6.98131165,  36.6750953 ],
                           [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.50404659,   4.40624189,  49.8360717 ],
                           [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.14951605,   9.1189445 ,  57.24997635],
                           [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.22649103,   3.55630312,  75.65716287]])
        self.confirm( i_got, i_exp,5 )


    def test83Elements(self):
        """Portfolio valuation elements, options"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.zeros(3)
        bondPositions = np.zeros(3)
        optionPositions = np.array([10,5,20.])
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.66452994e+00,   2.41069763e+01,   5.73791293e+02],
                           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.26562458e+00,   2.96525141e+01,   7.05068124e+02],
                           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.43249160e+00,   3.49065583e+01,   7.33501906e+02],
                           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   5.04046587e+00,   2.20312095e+01,   9.96721434e+02],
                           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.49516053e+00,   4.55947225e+01,   1.14499953e+03],
                           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.26491029e+00,   1.77815156e+01,   1.51314326e+03]])
        self.confirm( i_got, i_exp,5)



    def test85Elements(self):
        """Portfolio valuation elements, bonds all size 1"""
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.zeros(3)
        bondPositions = np.ones(3)
        optionPositions = np.zeros(3)
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([   [ 0.        ,  0.        ,  0.        ,  0.89577771,  0.73662811,  0.88781048,  0.        ,  0.        ,  0.        ],
                                       [ 0.        ,  0.        ,  0.        ,  0.89350822,  0.75013747,  0.88622697,  0.        ,  0.        ,  0.        ],
                                       [ 0.        ,  0.        ,  0.        ,  0.89282865,  0.73863486,  0.88481536,  0.        ,  0.        ,  0.        ],
                                       [ 0.        ,  0.        ,  0.        ,  0.89529413,  0.7495132 ,  0.88681669,  0.        ,  0.        ,  0.        ],
                                       [ 0.        ,  0.        ,  0.        ,  0.89689834,  0.7390318 ,  0.88825342,  0.        ,  0.        ,  0.        ],
                                       [ 0.        ,  0.        ,  0.        ,  0.89670981,  0.74914517,  0.88846263,  0.        ,  0.        ,  0.        ]])

        self.confirm( i_got, i_exp,5)


    def test86Elements(self):
        """Portfolio valuation elements, bonds """
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.zeros(3)
        bondPositions = np.array([15,5,20.])
        optionPositions = np.zeros(3)
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([   [  0.        ,   0.        ,   0.        ,  13.43666564,   3.68314053,  17.75620968,   0.        ,   0.        ,   0.        ],
                                       [  0.        ,   0.        ,   0.        ,  13.4026233 ,   3.75068737,  17.72453931,   0.        ,   0.        ,   0.        ],
                                       [  0.        ,   0.        ,   0.        ,  13.39242978,   3.69317429,  17.69630713,   0.        ,   0.        ,   0.        ],
                                       [  0.        ,   0.        ,   0.        ,  13.42941193,   3.74756602,  17.73633381,   0.        ,   0.        ,   0.        ],
                                       [  0.        ,   0.        ,   0.        ,  13.45347506,   3.69515899,  17.76506835,   0.        ,   0.        ,   0.        ],
                                       [  0.        ,   0.        ,   0.        ,  13.45064712,   3.74572583,  17.7692525 ,   0.        ,   0.        ,   0.        ]])
        self.confirm( i_got, i_exp,5)




    def test88Elements(self):
        """Portfolio valuation elements """
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.array([1,5,10.])
        bondPositions = np.array([7,5,20.])
        optionPositions = np.array([15,2,30.])
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([[  1.06636477e+01,   3.64438129e+02,   2.52113529e+03,   6.27044397e+00,   3.68314053e+00,   1.77562097e+01,   3.99679490e+00,   9.64279052e+00,   8.60686940e+02],
                                   [  1.22475605e+01,   3.13464449e+02,   2.61010292e+03,   6.25455754e+00,   3.75068737e+00,   1.77245393e+01,   1.89843687e+00,   1.18610056e+01,   1.05760219e+03],
                                   [  9.94966068e+00,   3.07457650e+02,   2.61312066e+03,   6.24980056e+00,   3.69317429e+00,   1.76963071e+01,   6.64873740e+00,   1.39626233e+01,   1.10025286e+03],
                                   [  9.57379580e+00,   3.48682835e+02,   2.78663272e+03,   6.26705890e+00,   3.74756602e+00,   1.77363338e+01,   7.56069880e+00,   8.81248378e+00,   1.49508215e+03],
                                   [  1.13481144e+01,   2.73486858e+02,   2.89034073e+03,   6.27828836e+00,   3.69515899e+00,   1.77650683e+01,   2.24274079e+00,   1.82378890e+01,   1.71749929e+03],
                                   [  1.10851403e+01,   3.79563570e+02,   3.10537682e+03,   6.27696866e+00,   3.74572583e+00,   1.77692525e+01,   3.39736544e+00,   7.11260624e+00,   2.26971489e+03]])
        self.confirm( i_got, i_exp,4)



    def test89Elements(self):
        """Portfolio valuation elements, with defaults """
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        samples[1:4,1] = 1e-9
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.array([1,5,10.])
        bondPositions = np.array([7,5,20.])
        optionPositions = np.array([15,2,30.])
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValueElements(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([[  1.06636477e+01,   3.64438129e+02,   2.52113529e+03,   6.27044397e+00,   3.68314053e+00,   1.77562097e+01,   3.99679490e+00,   9.64279052e+00,   8.60686940e+02],
                                   [  1.22086422e+01,   0.00000000e+00,   2.57380612e+03,   6.25497887e+00,   7.50000000e-01,   1.77298757e+01,   1.93327985e+00,   1.08865697e+02,   9.75554677e+02],
                                   [  9.92408516e+00,   0.00000000e+00,   2.58369472e+03,   6.25014596e+00,   7.50000000e-01,   1.77007939e+01,   6.73989623e+00,   1.08865697e+02,   1.03352844e+03],
                                   [  9.53608532e+00,   0.00000000e+00,   2.73865919e+03,   6.26756260e+00,   7.50000000e-01,   1.77428375e+01,   7.72755958e+00,   1.08865697e+02,   1.37552297e+03],
                                   [  1.13481144e+01,   2.73486858e+02,   2.89034073e+03,   6.27828836e+00,   3.69515899e+00,   1.77650683e+01,   2.24274079e+00,   1.82378890e+01,   1.71749929e+03],
                                   [  1.10851403e+01,   3.79563570e+02,   3.10537682e+03,   6.27696866e+00,   3.74572583e+00,   1.77692525e+01,   3.39736544e+00,   7.11260624e+00,   2.26971489e+03]])
        self.confirm( i_got, i_exp,3)



    def test95Value(self):
        """Portfolio valuation elements, with defaults """
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6,3*4)),axis=0), axis=1) ))
        samples[1:4,1] = 1e-9
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.array([1,5,10.])
        bondPositions = np.array([7,5,20.])
        optionPositions = np.array([15,2,30.])
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        i_got = portfolioValue(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        i_exp = np.array([ 3798.27338989,  3697.10327265,  
                                    3767.45378099,  4265.07190245,  
                                    4940.89413644,  5804.04233231])
        self.confirm( i_got, i_exp,3)



    def test99Speed(self):
        """Code performance """
        samples = np.abs(np.sin( np.cumsum( np.cumsum( np.ones((6000,3*4)),axis=0), axis=1) ))
        samples[1:4,1] = 1e-9
        corrEq = np.array([[1,0.11,0.22],[0.11,1,0.33],[0.22,0.33,1]])
        corrVol = np.array([[1,0.101,0.202],[0.101,1,0.303],[0.202,0.303,1]])
        corrHzd = np.array([[1,0.01,0.02],[0.01,1,0.03],[0.02,0.03,1]])
        T = 1.5/52.
        S0 = np.array([10., 60., 250.])
        h0 = np.array([0.01, 0.02, 0.015])
        sigma0 = np.array([0.5,0.6,0.4])
        muE = np.array([0.01, 0.02, 0.03])
        muH = np.array([0.0, 0, 0])
        eta = np.array([0.0, 0, 0])
        sigmaH = np.array([0.2,0.11,0.22])
        varvol = np.array([0.32,0.31,0.33])
        bondTenors = np.array([5,7,2.5])
        bondRiskFreeRates = np.array([0.012, 0.022, 0.033])
        recoveryRates = np.array([0.1, 0.15, 0.3])
        callput = np.array([-1,-1,1])
        K = np.array([9., 55., 240.])
        r = np.array([0.012, 0.022, 0.033])
        optionTenors = np.array([0.21, 0.5, 0.3])
        q = np.array([0.0012, 0.0022, 0.0033])
        eqPositions = np.array([1,5,10.])
        bondPositions = np.array([7,5,20.])
        optionPositions = np.array([15,2,30.])
        rhoDefault = 0.45
        rhoSame = 0.1
        rhoUnrelated = 0.05
        startTime = datetime.datetime.now()
        result = portfolioValue(T,
                    eqPositions, bondPositions, optionPositions,
                    S0, h0, sigma0,
                    muE, muH, eta,
                    sigmaH, varvol,
                    bondTenors, bondRiskFreeRates, recoveryRates,
                    callput, K, r, optionTenors, q,
                    rhoDefault,
                    corrEq, corrVol, corrHzd,
                    rhoSame, rhoUnrelated,
                    samples)
        endTime = datetime.datetime.now()
        self.failUnless( (endTime-startTime)<datetime.timedelta(seconds=5.0) )


    def testA11ExpShortfall(self):
        """
        Expected shortfall, trivial "
        """
        specialOffset = 0.12345
        simValues = np.cast[np.float64](specialOffset+np.arange(0,40,4))
        levl = 0.11
        i_got = expectedShortfall(simValues, level=levl)
        i_exp = specialOffset
        self.confirm( i_got, i_exp)

    def testA22ExpShortfall(self):
        """
        Expected shortfall, trivial with weights
        """
        specialOffset = 0.12345
        simValues = np.cast[np.float64](specialOffset+np.arange(0,40,2))
        levl = 0.05
        weights = 0.15*(1.0+np.sin(np.r_[:20]))
        i_got = expectedShortfall(simValues, level=levl, weights=weights)
        i_exp = 3.446418241465222
        self.confirm( i_got, i_exp)

    def testA33ExpShortfall(self):
        """
        Expected shortfall, nontrivial "
        """
        specialOffset = 1.0
        simValues = np.cast[np.float64](specialOffset+np.sin(np.r_[:20]))
        levl = 0.11
        i_got = expectedShortfall(simValues, level=levl)
        i_exp = 0.019306150784869858
        self.confirm( i_got, i_exp)

    def testA44ExpShortfall(self):
        """
        Expected shortfall, precise boundary "
        """
        i_got = expectedShortfall(1000+1000*np.sin(np.r_[:20]))
        i_exp = 0.0097934492965805475
        self.confirm( i_got, i_exp)



def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPortfolioValue)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())