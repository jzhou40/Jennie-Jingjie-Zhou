from __future__ import division
import string, unittest, datetime
import copy as pycopy
from numpy import *
from pprint import pprint
from MonteCarloPricing import MCRangePayoff, MCRange, MCRangeValues
import scipy.stats 

class TestMCRange(unittest.TestCase):
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

    def test11_RangePayoff_Z(self):
        'zero path val'
        path=array([2.1,5,10,6,])
        r = 0.05
        t= 1.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=1
        KHigh=2
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        result = MCRangePayoff(path, r, t, fixingTimes, finalT, fixingValues, KLow, KHigh, coupon)
        expected = 0.
        self.confirm( result, expected )

    def test14_RangePayoff_fullZtZ(self):
        'nonzero path val and nonzero int rate zero t'
        path=array([2.1,5,10,6,])
        r = 0.03
        t = 0.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        result = MCRangePayoff(path, r, t, fixingTimes, finalT, fixingValues, KLow, KHigh, coupon)
        expected = 0.081368329710864593
        self.confirm( result, expected )

    def test13_RangePayoff_fullZ(self):
        'nonzero path val and zero int rate'
        path=array([2.1,5,10,6,])
        r = 0.0
        t = 1.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        result = MCRangePayoff(path, r, t, fixingTimes, finalT, fixingValues, KLow, KHigh, coupon)
        expected = 0.09
        self.confirm( result, expected )


    def test15_RangePayoff_full(self):
        'nonzero path val and int rate'
        path=array([2.1,5,10,6,])
        r = 0.05
        t = 1.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        result = MCRangePayoff(path, r, t, fixingTimes, finalT, fixingValues, KLow, KHigh, coupon)
        expected = 0.079995404489847419
        self.confirm( result, expected )

        
    def test32_RangeV_short_fwd(self):
        'Range vals fwd integration, short'
        r = 0.05
        t = 1.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        sigma=0.8
        q=0.011
        samples = array([[0.5,0.5,0.5,0.5],[0.2,0.4,0.4,0.2],[0.6,0.6,0.7,0.6]])
        result = MCRangeValues(S0,
            KLow, KHigh, coupon,
            sigma, r, t, q,
            fixingTimes, finalT, fixingValues, samples=samples,
            integrationType='strong')
        expected = {'Samples':array([ 0.06116011,  0.06639676,  0.        ])}
        self.confirm( result, expected )

        
    def test32b_RangeV_short_fwd_milstein(self):
        'Range vals fwd integration, short, milstein'
        r = 0.05
        t = 1.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        sigma=0.8
        q=0.011
        samples = array([[0.5,0.5,0.5,0.5],[0.2,0.4,0.4,0.2],[0.6,0.6,0.7,0.6]])
        result = MCRangeValues(S0,
            KLow, KHigh, coupon,
            sigma, r, t, q,
            fixingTimes, finalT, fixingValues, samples=samples,
            integrationType='milstein')
        expected = {'Samples':array([ 0.13354711,  0.04756147,  0.        ])}
        self.confirm( result, expected )

        
    def test32c_RangeV_short_fwd_euler(self):
        'Range vals fwd integration, short, euler'
        r = 0.05
        t = 1.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        sigma=0.8
        q=0.011
        samples = array([[0.5,0.5,0.5,0.5],[0.2,0.4,0.4,0.2],[0.6,0.6,0.7,0.6]])
        result = MCRangeValues(S0,
            KLow, KHigh, coupon,
            sigma, r, t, q,
            fixingTimes, finalT, fixingValues, samples=samples,
            integrationType='euler')
        expected = {'Samples':array([ 0.        ,  0.13878376,  0.        ])}
        self.confirm( result, expected )

        
    def test32c_RangeV_short_fwd_imp(self):
        'Range vals fwd integration, short, importance sampled'
        r = 0.05
        t = 1.
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        shift = -0.3
        finalT = fixingTimes[-1]+0.7
        coupon=0.1
        sigma=0.8
        q=0.011
        samples = array([[0.5,0.5,0.5,0.5],[0.2,0.4,0.4,0.2],[0.6,0.6,0.7,0.6]])
        result = MCRangeValues(S0,
            KLow, KHigh, coupon,
            sigma, r, t, q,
            fixingTimes, finalT, fixingValues, samples=samples,shift=shift)
        expected = { 'Weights': array([ 0.98681272,  0.81546842,  1.07343172]), 
                        'Samples': array([ 0.1523824 ,  0.04756147,  0.        ])}
        self.confirm( result, expected )

        
    def test34_RangeV_long_fwd(self):
        'Range vals fwd integration'
        r = 0.05
        t = 0.075
        S0=10
        fixingTimes = array([1.5, 2, 2.2, 3])
        finalT = 5
        fixingValues = []
        KLow=4
        KHigh=7
        coupon=0.1
        sigma=0.8
        q=0.011
        fixingTimes = array([ 0.1       ,  0.2841471 ,  0.47507684,  0.58918884,  0.61350859,
                                    0.61761616,  0.68967462,  0.85537328,  1.0543091 ,  1.19552095,
                                    1.24111884,  1.24111982,  1.28746252,  1.42947923,  1.62853996,
                                    1.79356875,  1.86477842,  1.86863867,  1.89353994,  2.00852766,
                                    2.19982219,  2.38348775,  2.48260262,  2.49798058,  2.50742274,
                                    2.59418757,  2.77044341,  2.96608101,  3.09317159,  3.1268082 ])
        finalT = fixingTimes[-1]+0.7
        samples=array([
                       [ 0.68107334,  0.18005047,  0.82580041,  0.94821092,  0.81641089,
                         0.13604805,  0.32119733,  0.360656  ,  0.36271183,  0.28194748,
                         0.72807769,  0.78528557,  0.29102998,  0.72212716,  0.41350893,
                         0.98798562,  0.05548167,  0.83485874,  0.80438306,  0.49644389,
                         0.10590318,  0.44485515,  0.27812285,  0.42264046,  0.44994646,
                         0.30915757,  0.11533477,  0.85791568,  0.85584739,  0.00941799],
                        [ 0.44031527,  0.90368227,  0.35891744,  0.86764446,  0.65285961,
                         0.22879985,  0.69746302,  0.29524951,  0.30230741,  0.9413651 ,
                         0.65010541,  0.40709952,  0.75067071,  0.36495318,  0.18265871,
                         0.96065698,  0.41524042,  0.55038752,  0.21702991,  0.08408498,
                         0.43286517,  0.9765747 ,  0.42472427,  0.46201037,  0.71585553,
                         0.18614315,  0.27345213,  0.21306483,  0.33426945,  0.32498489],
                       [ 0.89430472,  0.00933838,  0.30305435,  0.98897502,  0.00729309,
                         0.44385447,  0.74542709,  0.21354469,  0.35307626,  0.32116135,
                         0.66136985,  0.36824785,  0.09670786,  0.62890469,  0.42769116,
                         0.34916683,  0.21371499,  0.41952978,  0.21828223,  0.48906868,
                         0.55383053,  0.54725378,  0.97676798,  0.78753179,  0.90193246,
                         0.25971425,  0.65468324,  0.54972182,  0.59068416,  0.88424866],
                       [ 0.45077413,  0.47561991,  0.65583086,  0.87690634,  0.32524024,
                         0.23773561,  0.9585397 ,  0.1560175 ,  0.02684398,  0.94184233,
                         0.25477609,  0.72281559,  0.40446686,  0.03396382,  0.16792473,
                         0.74706617,  0.80085372,  0.45631738,  0.55793407,  0.54309282,
                         0.44107821,  0.64594173,  0.36610688,  0.10516922,  0.05993047,
                         0.2826601 ,  0.71987593,  0.38065444,  0.56928809,  0.68506633],
                       [ 0.16284022,  0.57591365,  0.92252267,  0.62273133,  0.36803241,
                         0.82805673,  0.6300896 ,  0.54014246,  0.5645388 ,  0.55042009,
                         0.16656471,  0.6132116 ,  0.18432316,  0.34205857,  0.86173871,
                         0.24802175,  0.4696679 ,  0.86859215,  0.40940975,  0.80244362,
                         0.07238344,  0.67177761,  0.96833386,  0.65001485,  0.44394698,
                         0.52532018,  0.5778596 ,  0.13099623,  0.30257245,  0.63912968],
                       [ 0.446786  ,  0.73155878,  0.39175213,  0.15589475,  0.85537663,
                         0.39020788,  0.20751555,  0.93170214,  0.06898427,  0.25336347,
                         0.43870915,  0.63934604,  0.33470618,  0.24246442,  0.21770289,
                         0.95363234,  0.61179029,  0.1728418 ,  0.97766006,  0.90743766,
                         0.71615285,  0.48092158,  0.03893334,  0.84030962,  0.41979723,
                         0.77540633,  0.29210904,  0.43855248,  0.82836678,  0.37000148],
                       [ 0.00472328,  0.95614847,  0.82135788,  0.87035665,  0.02787566,
                         0.14034911,  0.45730374,  0.10196142,  0.97080544,  0.92333065,
                         0.50827906,  0.53967063,  0.07948539,  0.2747325 ,  0.17706795,
                         0.71950587,  0.37428241,  0.64701754,  0.44041335,  0.17443951,
                         0.34398311,  0.0091408 ,  0.7824831 ,  0.84511481,  0.58475421,
                         0.26008784,  0.86522673,  0.10443811,  0.53012914,  0.67157359]])

        result = MCRangeValues(S0,
            KLow, KHigh, coupon,
            sigma, r, t, q,
            fixingTimes, finalT, fixingValues, samples=samples,
            integrationType='strong')
        expected = {'Samples':array([ 0.04785517,  0.06091443,  0.06098108, 
                                    0.01335179,  0.07727881,
                                    0.18195948,  0.03329502])}
        self.confirm( result, expected )

        

        

def single():
    suite = unittest.TestSuite()
    suite.addTest(TestMCRange('test33_RangeV_short_bridge'))
    return suite

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMCRange)
    return suite
    
if __name__ == '__main__':
    tests=suite()
    #tests=single()
    unittest.TextTestRunner(verbosity=2).run(tests)