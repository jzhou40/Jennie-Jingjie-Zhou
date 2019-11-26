from __future__ import division
import string, unittest, datetime
import copy as pycopy
from numpy import *
from pprint import pprint
from BSPaths import  MakeBSPaths
import scipy.stats 

class TestBSPaths(unittest.TestCase):
    def confirm( self, result, expected, digits=7, logOnErr=True):
        res = pycopy.copy( result )
        try:
            self.confirm_iter( res, expected, digits=digits)
        except:
            if logOnErr:
                print('Did Not Match Expected:\n\t\t%r'%(
                                result,))
            raise
    
    def confirm_iter( self, result, expected, digits=7):
        ''' Nest iterables and check.  Newer unittest in Python v2.7 may
           supersede the necessity of this function.
        '''
        if hasattr(expected,'__iter__'):
            if type(expected)==dict:
                for (key,expectedElem) in expected.iteritems():
                    resElem = result[key]
                    self.confirm_iter(resElem, expectedElem, digits )
            else:
                for (i, expectedElem) in enumerate(expected):
                    resElem = result[i]
                    self.confirm_iter(resElem, expectedElem, digits )
        elif expected is None:
            self.failUnless( result is None )
        elif isinstance(expected, datetime.datetime) or isinstance(expected, str):
            self.assertEqual( result, expected )
        else:
            self.assertAlmostEqual(result, expected, digits)


    def test61_MakeBSPaths_Standard_Triv(self):
        'MakeBSPaths: standard, trivial'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[0.25], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')
        expected = {'Paths':array([[ 10.        ],
                                   [  2.83128399],
                                   [ 35.3196643 ]])}
        self.confirm( result, expected )

    def test62_MakeBSPaths_Standard_nzT(self):
        'MakeBSPaths: standard, trivial, nonzero start time'
        S0 = 10
        sigma = 0.8
        r = 0.5*0.64
        q = 0.0
        t = 0.5
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[0.25], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 10.        ],
                                       [  3.92742769],
                                       [ 25.46195828]])                                 
        self.confirm( result, expected )

    def test63_MakeBSPaths_Standard_nzq(self):
        'MakeBSPaths: standard, trivial, nonzero q'
        S0 = 10
        sigma = 1.
        r = 0.6
        q = 0.1
        t = 0
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[0.25], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 10.        ],
                                   [  2.83128399],
                                   [ 35.3196643 ]])
        self.confirm( result, expected )

    def test64_MakeBSPaths_Standard_drifting(self):
        'MakeBSPaths: standard, trivial, drifting'
        S0 = 10
        sigma = 1.0
        r = 0.045
        q = 0.01
        t = 0
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[0.25], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 1.96420011],
                                   [ 0.55612083],
                                   [ 6.93748884]])
        self.confirm( result, expected )

    def test71_MakeBSPaths_Standard_NoDrift_2Fixings(self):
        'MakeBSPaths: standard, 2 fixings, no drift'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0
        fixingTimes = array( [1,3.5,],ndmin=1)
        samples=array([[0.5,0.5],[0.25,0.5], [0.5,0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 10.        ,  10.        ],
                                   [  5.09416284,   5.09416284],
                                   [ 10.        ,  29.05082922]])
        self.confirm( result, expected )

    def test72_MakeBSPaths_Standard_Drift_2Fixings(self):
        'MakeBSPaths: standard, 2 fixings, with drift'
        S0 = 10
        sigma = 1.0
        r = 0.055
        q = 0.01
        t = 0
        fixingTimes = array( [1,3.5,],ndmin=1)
        samples=array([[0.5,0.5],[0.25,0.5], [0.5,0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 6.34447968,  2.03416434],
                                   [ 3.23198126,  1.03623644],
                                   [ 6.34447968,  5.9094161 ]])
        self.confirm( result, expected )

    def test73_MakeBSPaths_Standard_NoDrift_2Fixings_nzt(self):
        'MakeBSPaths: standard, 2 fixings, nonzero start'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0.5
        fixingTimes = array( [1,3.5,],ndmin=1)
        samples=array([[0.5,0.5],[0.25,0.5], [0.5,0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 10.        ,  10.        ],
                                   [  6.2068208 ,   6.2068208 ],
                                   [ 10.        ,  29.05082922]])
        self.confirm( result, expected )

    def test74_MakeBSPaths_Standard_NoDrift_3Fixings(self):
        'MakeBSPaths: standard, 3 fixings, no drift'
        S0 = 10
        sigma = .8
        r = 0.5*0.64
        q = 0.0
        t = 0
        fixingTimes = array( [1,1.1,3.5,],ndmin=1)
        samples=array([[0.5,0.5,0.5],[0.25,0.5,0.25], [0.5,0.75,0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 10.        ,  10.        ,  10.        ],
                                   [  5.82986179,   5.82986179,   2.52707187],
                                   [ 10.        ,  11.86056466,  27.36188605]])
        self.confirm( result, expected )

    def test75_MakeBSPaths_Standard_Drift_30Fixings(self):
        'MakeBSPaths: standard, 30 fixings, with drift and nonzero start'
        S0 = 10
        sigma = 0.8
        r = 0.05
        q = 0.01
        t = 0.075
        fixingTimes = array([ 0.1       ,  0.2841471 ,  0.47507684,  0.58918884,  0.61350859,
                                    0.61761616,  0.68967462,  0.85537328,  1.0543091 ,  1.19552095,
                                    1.24111884,  1.24111982,  1.28746252,  1.42947923,  1.62853996,
                                    1.79356875,  1.86477842,  1.86863867,  1.89353994,  2.00852766,
                                    2.19982219,  2.38348775,  2.48260262,  2.49798058,  2.50742274,
                                    2.59418757,  2.77044341,  2.96608101,  3.09317159,  3.1268082 ])
        samples=array([[  8.05099964e-01,   5.91286873e-01,   4.27881171e-01,
                              6.06069662e-02,   9.55215373e-01,   6.38195742e-01,
                              7.10052128e-01,   5.62244453e-02,   9.64285117e-01,
                              4.87304166e-01,   5.70182961e-01,   2.27806447e-01,
                              1.36414491e-01,   9.62507742e-01,   1.69067662e-01,
                              1.37740922e-01,   7.19056665e-01,   2.50074387e-01,
                              6.03244885e-01,   1.82785613e-01,   6.16149935e-01,
                              1.57439959e-01,   6.84581905e-01,   6.84919480e-01,
                              5.92628671e-01,   8.72800885e-01,   6.20408079e-01,
                              9.11836403e-01,   6.47070704e-01,   9.37190812e-01],
                           [  6.87752141e-01,   3.74382160e-01,   2.26729966e-01,
                              5.22109491e-01,   9.48758471e-01,   9.13398302e-01,
                              8.68590945e-01,   5.71660388e-01,   1.76460918e-01,
                              5.54320755e-02,   2.04333090e-01,   4.29724704e-01,
                              8.80908652e-01,   9.36234266e-01,   7.30450799e-01,
                              8.05612366e-01,   6.58953734e-01,   2.50710754e-01,
                              1.13209314e-01,   9.33318306e-01,   4.75295081e-01,
                              5.83819664e-02,   2.26961074e-01,   8.13245303e-01,
                              7.93957298e-01,   6.07539567e-01,   3.09103788e-02,
                              6.73086450e-01,   9.13338923e-01,   5.81791353e-01],
                           [  3.82342579e-01,   9.49515373e-01,   1.39786064e-01,
                              4.75652540e-01,   8.32982887e-01,   3.97892597e-01,
                              6.22597795e-01,   9.16328487e-02,   8.10651202e-05,
                              9.92464216e-01,   3.18339127e-01,   8.69973623e-01,
                              8.88780897e-01,   8.88516010e-01,   8.42314275e-01,
                              4.01166680e-01,   8.80230560e-01,   5.68504059e-01,
                              6.18173025e-01,   4.87502698e-01,   5.97926414e-01,
                              6.43379139e-01,   5.30892151e-02,   6.77560627e-02,
                              4.94126587e-01,   6.91132102e-01,   4.76739279e-01,
                              7.19110780e-01,   4.55150138e-01,   1.96220515e-01]])
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong')['Paths']
        expected = array([[ 11.07139676,  11.38226595,  10.12548531,   6.45146928,
                                  7.91925688,   8.05488861,   8.89042364,   5.06160051,
                                  9.10868884,   8.67218538,   8.82477285,   8.81955765,
                                  7.20766888,  11.84831106,   7.96105046,   5.33315623,
                                  5.91701688,   5.71582047,   5.8668024 ,   4.44446785,
                                  4.67136861,   3.14385474,   3.45122398,   3.60452853,
                                  3.66109327,   4.6741246 ,   4.93153293,   7.53324901,
                                  8.09609346,  10.0410825 ],
                               [ 10.56451841,   8.98891214,   6.5566353 ,   6.44632923,
                                  7.84929197,   8.40734004,  10.47923631,  10.61017367,
                                  7.20409869,   4.28798762,   3.67628816,   3.67577164,
                                  4.44565119,   6.76384159,   7.96512033,  10.06388678,
                                 10.76670613,  10.4016401 ,   8.86654657,  12.90055021,
                                 11.96547998,   6.63818651,   5.34665193,   5.81505543,
                                  6.18154579,   6.4339271 ,   3.27063232,   3.62875058,
                                  5.16358244,   5.27251423],
                               [  9.56128368,  15.94635584,  10.35835061,   9.86840678,
                                 11.05682535,  10.89853356,  11.42176209,   7.06930515,
                                  1.74062751,   3.47440324,   3.16440299,   3.16722589,
                                  3.85742258,   5.35291562,   7.24467841,   6.37704556,
                                  8.03534487,   8.09580835,   8.35059682,   8.01758263,
                                  8.28828757,   8.92999934,   5.78203774,   4.96475375,
                                  4.94598125,   5.42969868,   5.0679586 ,   5.89121691,
                                  5.50549595,   4.81074454]])
        self.confirm( result, expected )

    def test81_MakeBSPaths_Imp_Triv(self):
        'MakeBSPaths: importance, trivial'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[scipy.stats.distributions.norm.cdf(-1.666666)], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong', shift=1.666666)
        expected = {'Paths': array([[ 226.02184821],
                                   [  10.        ],
                                   [ 798.30158023]]),
                    'Weights': array([ 0.24935249,  4.01038713,  0.08102181])}
        self.confirm( result, expected )
        
    def test81b_MakeBSPaths_Imp_Triv(self):
        'MakeBSPaths: importance (-), trivial'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[scipy.stats.distributions.norm.cdf(1.666666)], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong', shift=-1.666666)
        expected = {'Paths': array([[ 0.4424351 ],
                                   [10.        ],
                                   [ 1.56266594]]),
                    'Weights': array([ 0.24935249,  4.01038713,  0.76740649])}
        self.confirm( result, expected )

    def test82_MakeBSPaths_Imp_NoDrift_2Fixings_nzt(self):
        'MakeBSPaths: importance, 2 fixings, nonzero start'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0.5
        fixingTimes = array( [1,3.5,],ndmin=1)
        samples=array([[0.5,0.5],[0.25,0.5], [0.5,0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='strong', shift=1.666666)
        expected = {'Paths': array([[  12.17036094,  109.40196367],
                                   [   7.55392495,   67.90383839],
                                   [  12.17036094,  317.8217763 ]]), 
                    'Weights': array([ 0.36674606,  0.44231746,  0.14372169])}
        self.confirm( result, expected )

    def test75_MakeBSPaths_Imp_Drift_30Fixings(self):
        'MakeBSPaths: importance, 30 fixings, with drift and nonzero start'
        S0 = 10
        sigma = 0.8
        r = 0.05
        q = 0.01
        t = 0.075
        fixingTimes = array([ 0.1       ,  0.2841471 ,  0.47507684,  0.58918884,  0.61350859,
                                    0.61761616,  0.68967462,  0.85537328,  1.0543091 ,  1.19552095,
                                    1.24111884,  1.24111982,  1.28746252,  1.42947923,  1.62853996,
                                    1.79356875,  1.86477842,  1.86863867,  1.89353994,  2.00852766,
                                    2.19982219,  2.38348775,  2.48260262,  2.49798058,  2.50742274,
                                    2.59418757,  2.77044341,  2.96608101,  3.09317159,  3.1268082 ])
        samples=array([[  8.05099964e-01,   5.91286873e-01,   4.27881171e-01,
                              6.06069662e-02,   9.55215373e-01,   6.38195742e-01,
                              7.10052128e-01,   5.62244453e-02,   9.64285117e-01,
                              4.87304166e-01,   5.70182961e-01,   2.27806447e-01,
                              1.36414491e-01,   9.62507742e-01,   1.69067662e-01,
                              1.37740922e-01,   7.19056665e-01,   2.50074387e-01,
                              6.03244885e-01,   1.82785613e-01,   6.16149935e-01,
                              1.57439959e-01,   6.84581905e-01,   6.84919480e-01,
                              5.92628671e-01,   8.72800885e-01,   6.20408079e-01,
                              9.11836403e-01,   6.47070704e-01,   9.37190812e-01],
                           [  6.87752141e-01,   3.74382160e-01,   2.26729966e-01,
                              5.22109491e-01,   9.48758471e-01,   9.13398302e-01,
                              8.68590945e-01,   5.71660388e-01,   1.76460918e-01,
                              5.54320755e-02,   2.04333090e-01,   4.29724704e-01,
                              8.80908652e-01,   9.36234266e-01,   7.30450799e-01,
                              8.05612366e-01,   6.58953734e-01,   2.50710754e-01,
                              1.13209314e-01,   9.33318306e-01,   4.75295081e-01,
                              5.83819664e-02,   2.26961074e-01,   8.13245303e-01,
                              7.93957298e-01,   6.07539567e-01,   3.09103788e-02,
                              6.73086450e-01,   9.13338923e-01,   5.81791353e-01],
                           [  3.82342579e-01,   9.49515373e-01,   1.39786064e-01,
                              4.75652540e-01,   8.32982887e-01,   3.97892597e-01,
                              6.22597795e-01,   9.16328487e-02,   8.10651202e-05,
                              9.92464216e-01,   3.18339127e-01,   8.69973623e-01,
                              8.88780897e-01,   8.88516010e-01,   8.42314275e-01,
                              4.01166680e-01,   8.80230560e-01,   5.68504059e-01,
                              6.18173025e-01,   4.87502698e-01,   5.97926414e-01,
                              6.43379139e-01,   5.30892151e-02,   6.77560627e-02,
                              4.94126587e-01,   6.91132102e-01,   4.76739279e-01,
                              7.19110780e-01,   4.55150138e-01,   1.96220515e-01]])
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          shift=-0.333,
          integrationType='strong')
        expected = {'Paths': array([[ 11.06757719,  11.30012123,   9.97946874,   6.33707496,
                                  7.77626157,   7.90926249,   8.71496393,   4.93257787,
                                  8.80801705,   8.34716716,   8.48681948,   8.48180399,
                                  6.92561022,  11.33158654,   7.55505579,   5.03164558,
                                  5.57324521,   5.38362535,   5.52393733,   4.17050679,
                                  4.35152351,   2.90854401,   3.18422212,   3.32511279,
                                  3.37702226,   4.30184266,   4.50952595,   6.83676699,
                                  7.3185717 ,   9.07188385],
                               [ 10.56087371,   8.92404002,   6.46208404,   6.33202605,
                                  7.70756   ,   8.25534187,  10.2724201 ,  10.33971522,
                                  6.96629616,   4.12728141,   3.53550108,   3.53500434,
                                  4.27167894,   6.46885922,   7.5589181 ,   9.49492369,
                                 10.14117325,   9.797112  ,   8.34837179,  12.10534852,
                                 11.14621255,   6.1413326 ,   4.93301143,   5.36428413,
                                  5.70190821,   5.92148143,   2.9907539 ,   3.29325662,
                                  4.66768924,   4.76359364],
                               [  9.5579851 ,  15.83127252,  10.20897596,   9.69342499,
                                 10.85717604,  10.70149656,  11.19634437,   6.88910515,
                                  1.68317054,   3.34418873,   3.04321906,   3.04593386,
                                  3.70646955,   5.11946608,   6.87521701,   6.0165185 ,
                                  7.56850085,   7.62529182,   7.86257494,   7.52337151,
                                  7.72079474,   8.26160819,   5.33471388,   4.57989611,
                                  4.56221341,   4.9972372 ,   4.63427724,   5.34654799,
                                  4.97676652,   4.34639549]]),
                    'Weights': array([ 1.02516635,  0.97987709,  0.99601096])}
        self.confirm( result, expected )

    def test91a_MakeBSPaths_E_Triv(self):
        'MakeBSPaths: euler, trivial'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[scipy.stats.distributions.norm.cdf(-1.666666)], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='euler')['Paths']
        expected = array([ 27.5       ,  -3.68046575,  40.11854778]) # Note negative value!
        self.confirm( result, expected )
        
    def test91b_MakeBSPaths_M_Triv(self):
        'MakeBSPaths: milstein, trivial'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0
        fixingTimes = array( [3.5,],ndmin=1)
        samples=array([[0.5],[scipy.stats.distributions.norm.cdf(-1.666666)], [0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='milstein')['Paths']
        expected = array([ 10.        ,  27.43060647,  30.57993519]) 
        self.confirm( result, expected )
        

    def test82a_MakeBSPaths_E_NoDrift_2Fixings_nzt(self):
        'MakeBSPaths: euler, 2 fixings, nonzero start'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0.5
        fixingTimes = array( [1,3.5,],ndmin=1)
        samples=array([[0.5,0.5],[0.25,0.5], [0.5,0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='euler')['Paths']
        expected = array([[ 12.5       ,  28.125     ],
                           [  7.73063724,  17.39393379],
                           [ 12.5       ,  41.45577418]])
        self.confirm( result, expected )


    def test82b_MakeBSPaths_E_NoDrift_2Fixings_nzt(self):
        'MakeBSPaths: milstein, 2 fixings, nonzero start'
        S0 = 10
        sigma = 1.0
        r = 0.5
        q = 0.0
        t = 0.5
        fixingTimes = array( [1,3.5,],ndmin=1)
        samples=array([[0.5,0.5],[0.25,0.5], [0.5,0.75],],ndmin=2)
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='milstein')['Paths']
        expected = array([[ 10.        ,  10.        ],
                           [  6.3679783 ,   6.3679783 ],
                           [ 10.        ,  26.35132463]])
        self.confirm( result, expected )


    def test95a_MakeBSPaths_E_Drift_30Fixings(self):
        'MakeBSPaths: euler, 30 fixings, with drift and nonzero start'
        S0 = 10
        sigma = 0.8
        r = 0.05
        q = 0.01
        t = 0.075
        fixingTimes = array([ 0.1       ,  0.2841471 ,  0.47507684,  0.58918884,  0.61350859,
                                    0.61761616,  0.68967462,  0.85537328,  1.0543091 ,  1.19552095,
                                    1.24111884,  1.24111982,  1.28746252,  1.42947923,  1.62853996,
                                    1.79356875,  1.86477842,  1.86863867,  1.89353994,  2.00852766,
                                    2.19982219,  2.38348775,  2.48260262,  2.49798058,  2.50742274,
                                    2.59418757,  2.77044341,  2.96608101,  3.09317159,  3.1268082 ])
        samples=array([[  8.05099964e-01,   5.91286873e-01,   4.27881171e-01,
                              6.06069662e-02,   9.55215373e-01,   6.38195742e-01,
                              7.10052128e-01,   5.62244453e-02,   9.64285117e-01,
                              4.87304166e-01,   5.70182961e-01,   2.27806447e-01,
                              1.36414491e-01,   9.62507742e-01,   1.69067662e-01,
                              1.37740922e-01,   7.19056665e-01,   2.50074387e-01,
                              6.03244885e-01,   1.82785613e-01,   6.16149935e-01,
                              1.57439959e-01,   6.84581905e-01,   6.84919480e-01,
                              5.92628671e-01,   8.72800885e-01,   6.20408079e-01,
                              9.11836403e-01,   6.47070704e-01,   9.37190812e-01],
                           [  6.87752141e-01,   3.74382160e-01,   2.26729966e-01,
                              5.22109491e-01,   9.48758471e-01,   9.13398302e-01,
                              8.68590945e-01,   5.71660388e-01,   1.76460918e-01,
                              5.54320755e-02,   2.04333090e-01,   4.29724704e-01,
                              8.80908652e-01,   9.36234266e-01,   7.30450799e-01,
                              8.05612366e-01,   6.58953734e-01,   2.50710754e-01,
                              1.13209314e-01,   9.33318306e-01,   4.75295081e-01,
                              5.83819664e-02,   2.26961074e-01,   8.13245303e-01,
                              7.93957298e-01,   6.07539567e-01,   3.09103788e-02,
                              6.73086450e-01,   9.13338923e-01,   5.81791353e-01],
                           [  3.82342579e-01,   9.49515373e-01,   1.39786064e-01,
                              4.75652540e-01,   8.32982887e-01,   3.97892597e-01,
                              6.22597795e-01,   9.16328487e-02,   8.10651202e-05,
                              9.92464216e-01,   3.18339127e-01,   8.69973623e-01,
                              8.88780897e-01,   8.88516010e-01,   8.42314275e-01,
                              4.01166680e-01,   8.80230560e-01,   5.68504059e-01,
                              6.18173025e-01,   4.87502698e-01,   5.97926414e-01,
                              6.43379139e-01,   5.30892151e-02,   6.77560627e-02,
                              4.94126587e-01,   6.91132102e-01,   4.76739279e-01,
                              7.19110780e-01,   4.55150138e-01,   1.96220515e-01]])
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='euler')
        expected = {'Paths': array([[ 11.09779821,  12.05907488,  11.38493081,   6.66893039,
                                  8.08789067,   8.23586863,   9.2386204 ,   4.52444443,
                                  7.47078753,   7.44150083,   7.67987715,   7.67533962,
                                  6.2400782 ,   9.62522763,   6.41112409,   4.18128051,
                                  4.71094959,   4.55379568,   4.70880827,   3.57467515,
                                  3.97148625,   2.63218415,   2.96119673,   3.10446826,
                                  3.16218753,   4.0224443 ,   4.4649528 ,   6.63617263,
                                  7.38423001,   9.05356063],
                               [ 10.62915973,   9.53879755,   7.11196761,   7.25100045,
                                  8.73524251,   9.34667653,  11.62118441,  12.38168736,
                                  8.37616528,   4.40883818,   3.79458729,   3.7940553 ,
                                  4.57180985,   6.69820433,   8.21990556,  10.57647532,
                                 11.53144909,  11.14791504,   9.45666812,  13.35074107,
                                 13.16343181,   6.18136268,   5.03996483,   5.48802127,
                                  5.84002043,   6.23586564,   2.36838244,   2.76273965,
                                  3.84963237,   3.97143642],
                               [  9.63136902,  15.12544291,   9.52388947,   9.41018608,
                                 10.55344667,  10.41514113,  11.14369018,   6.38829061,
                                 -2.15832901,  -3.7476466 ,  -3.45208096,  -3.4551602 ,
                                 -4.1875602 ,  -5.74988658,  -7.85621511,  -7.26893898,
                                 -9.11475593,  -9.19434416,  -9.55250964,  -9.51525501,
                                -10.4136918 , -11.80231509,  -7.04665022,  -6.00746996,
                                 -6.00286326,  -6.72964797,  -6.64523343,  -8.06152221,
                                 -7.84348432,  -6.86986256]]),}
        self.confirm( result, expected )

    def test95b_MakeBSPaths_M_Drift_30Fixings(self):
        'MakeBSPaths: milstein, 30 fixings, with drift and nonzero start'
        S0 = 10
        sigma = 0.8
        r = 0.05
        q = 0.01
        t = 0.075
        fixingTimes = array([ 0.1       ,  0.2841471 ,  0.47507684,  0.58918884,  0.61350859,
                                    0.61761616,  0.68967462,  0.85537328,  1.0543091 ,  1.19552095,
                                    1.24111884,  1.24111982,  1.28746252,  1.42947923,  1.62853996,
                                    1.79356875,  1.86477842,  1.86863867,  1.89353994,  2.00852766,
                                    2.19982219,  2.38348775,  2.48260262,  2.49798058,  2.50742274,
                                    2.59418757,  2.77044341,  2.96608101,  3.09317159,  3.1268082 ])
        samples=array([[  8.05099964e-01,   5.91286873e-01,   4.27881171e-01,
                              6.06069662e-02,   9.55215373e-01,   6.38195742e-01,
                              7.10052128e-01,   5.62244453e-02,   9.64285117e-01,
                              4.87304166e-01,   5.70182961e-01,   2.27806447e-01,
                              1.36414491e-01,   9.62507742e-01,   1.69067662e-01,
                              1.37740922e-01,   7.19056665e-01,   2.50074387e-01,
                              6.03244885e-01,   1.82785613e-01,   6.16149935e-01,
                              1.57439959e-01,   6.84581905e-01,   6.84919480e-01,
                              5.92628671e-01,   8.72800885e-01,   6.20408079e-01,
                              9.11836403e-01,   6.47070704e-01,   9.37190812e-01],
                           [  6.87752141e-01,   3.74382160e-01,   2.26729966e-01,
                              5.22109491e-01,   9.48758471e-01,   9.13398302e-01,
                              8.68590945e-01,   5.71660388e-01,   1.76460918e-01,
                              5.54320755e-02,   2.04333090e-01,   4.29724704e-01,
                              8.80908652e-01,   9.36234266e-01,   7.30450799e-01,
                              8.05612366e-01,   6.58953734e-01,   2.50710754e-01,
                              1.13209314e-01,   9.33318306e-01,   4.75295081e-01,
                              5.83819664e-02,   2.26961074e-01,   8.13245303e-01,
                              7.93957298e-01,   6.07539567e-01,   3.09103788e-02,
                              6.73086450e-01,   9.13338923e-01,   5.81791353e-01],
                           [  3.82342579e-01,   9.49515373e-01,   1.39786064e-01,
                              4.75652540e-01,   8.32982887e-01,   3.97892597e-01,
                              6.22597795e-01,   9.16328487e-02,   8.10651202e-05,
                              9.92464216e-01,   3.18339127e-01,   8.69973623e-01,
                              8.88780897e-01,   8.88516010e-01,   8.42314275e-01,
                              4.01166680e-01,   8.80230560e-01,   5.68504059e-01,
                              6.18173025e-01,   4.87502698e-01,   5.97926414e-01,
                              6.43379139e-01,   5.30892151e-02,   6.77560627e-02,
                              4.94126587e-01,   6.91132102e-01,   4.76739279e-01,
                              7.19110780e-01,   4.55150138e-01,   1.96220515e-01]])
        fixingValues = array([])
        result = MakeBSPaths(S0, sigma, r, t, q,
          fixingTimes, fixingValues, samples,
          integrationType='milstein')
        expected = {'Paths': array([[ 11.07696345,  11.41848966,  10.10556575,   6.43671505,
                                  7.90054558,   8.0360099 ,   8.88590455,   5.06762203,
                                  9.09349337,   8.64734681,   8.80212003,   8.79691821,
                                  7.17833857,  11.78050959,   7.78478591,   5.15497138,
                                  5.73003867,   5.53502837,   5.68235897,   4.27582734,
                                  4.51156661,   2.99281047,   3.29389791,   3.44081537,
                                  3.4949625 ,   4.47476178,   4.7383625 ,   7.28824676,
                                  7.85562571,   9.74532123],
                               [ 10.56832767,   8.92532317,   6.41571742,   6.30758414,
                                  7.6805169 ,   8.22675568,  10.27688676,  10.42227397,
                                  6.95968248,   4.14816863,   3.55102825,   3.55052933,
                                  4.29896689,   6.55679422,   7.78625421,   9.91273049,
                                 10.61978941,  10.25938715,   8.74080714,  12.74307258,
                                 11.78722643,   6.54669915,   5.24665407,   5.70771394,
                                  6.06816085,   6.32353743,   3.28899575,   3.6721483 ,
                                  5.24434689,   5.35623826],
                               [  9.55853709,  15.9630626 ,  10.21629602,   9.72266076,
                                 10.89882963,  10.74263164,  11.27053901,   6.92172609,
                                  3.48918535,   6.83234919,   6.21605207,   6.22159729,
                                  7.58548439,  10.58278269,  14.46494686,  12.66763352,
                                 15.99499678,  16.11549198,  16.62646608,  15.95043509,
                                 16.54013613,  17.90487683,  11.6046324 ,   9.96341656,
                                  9.92567847,  10.92046397,  10.16964291,  11.91474504,
                                 11.11407959,   9.7023382 ]]),}
        self.confirm( result, expected )


def single():
    suite = unittest.TestSuite()
    suite.addTest(TestBSPaths('test81_MakeBSPaths_Imp_Triv'))
    return suite

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBSPaths)
    return suite
    
if __name__ == '__main__':
    tests=suite()
    # tests=single()
    unittest.TextTestRunner(verbosity=2).run(tests)