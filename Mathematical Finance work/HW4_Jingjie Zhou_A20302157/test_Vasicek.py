from __future__ import division
import string, unittest, datetime
import copy as pycopy
from numpy import *
from pprint import pprint
from Vasicek import  VasicekLimits, VasicekParams, VasicekDiagonals, \
                                CheckExercise, CallExercise, VasicekPolicyDiagonals,  \
                                Iterate, VasicekCallableZCBVals, VasicekCallableZCB, \
                                TridiagonalSolve
import scipy.stats 

class TestVasicek(unittest.TestCase):
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

    def test01_TDMA(self):
        ''' TDMA Solver 1 '''
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.99561612,
                            0.99342926,  0.99124749,  0.98907061,  0.98689808,  0.98472858])
        new = array([ 1.0048125 ,  1.0024076 ,  1.00000623,  0.99760999,  0.99521953,
                            0.99283505,  0.99045655,  0.98808379,  0.98571609,  0.98335189])
        got = TridiagonalSolve( subd,  d,  superd, old )
        expected = new
        self.confirm(got, expected)
        


    def test02_TDMA(self):
        ''' TDMA Solver 2 '''
        subd = array([ 0.  , -0.25, -0.15, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.55   ,  8.5002,  1.5004,  1.5006,  2.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.05, -0.25, -0.25, -0.17,  0.  ])
        old = array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.99561612,
                            0.99342926,  0.99124749,  0.98907061,  0.98689808,  0.98472858])
        new = array([ 1.00481251,  0.96321216,  0.76486494,  0.16421552,  0.82736191,
        0.81881521,  0.56888736,  0.90688886,  0.91979091,  0.98335189])
        got = TridiagonalSolve( subd,  d,  superd, old )
        expected = new
        self.confirm(got, expected)
        

    def test010_LimitsMiddle(self):
        'VasicekLimits: 50th pctile '
        r0 = 0.04
        sigma = 0.03
        kappa = 0.5
        theta = 0.07
        T = 5
        res = VasicekLimits(r0, sigma, kappa, theta,T,prob=0.5)
        expected=(0.067537450041283045 ,0.067537450041283045 )
        self.confirm(res, expected)
        
    def test011_Limits(self):
        'VasicekLimits: correct vals '
        r0 = 0.04
        sigma = 0.03
        kappa = 0.5
        theta = 0.07
        T = 5
        prob = 1e-3
        res = VasicekLimits(r0, sigma, kappa, theta,T,prob=prob)
        expected=(-0.024856663930153455,0.15993156401271955 )
        self.confirm(res, expected)
        
    def test012_Limits(self):
        'VasicekLimits: no mean rev '        
        r0 = 0.04
        sigma = 0.02
        kappa = 0.0
        theta = 0.07
        T = 5
        prob = 1e-2
        res = VasicekLimits(r0, sigma, kappa, theta,T,prob=prob)
        expected=(-0.06403743971334877, 0.14403743971334876)
        self.confirm(res, expected)
        
    def test021_Parms(self):
        'VasicekParams: N is integer'
        r0 = 0.04
        sigma = 0.03
        kappa = 0.5
        theta = 0.07
        T = 5
        prob = 1e-3
        M=250
        res =  VasicekParams(r0, M,sigma, kappa, theta,T,prob=prob)
        self.failUnless(type(res[2])==type(1))
        
    def test022_ParmsSimple(self):
        'VasicekParams:  kappa=0 gives stdev sigma*sqrt(t) '
        r0 = 0.04
        sigma = 0.03
        kappa = 0.
        theta = 0.0
        T = 5
        prob = 1e-3
        M=250
        res =  VasicekParams(r0, M,sigma, kappa, theta,T,prob=prob)
        expected=(-0.1672990850857152, 0.0086374618785714664, 49, 0.02)
        self.confirm(res, expected)
        
    def test023_Parms(self):
        'VasicekParams: correct vals '
        r0 = 0.04
        sigma = 0.03
        kappa = 0.5
        theta = 0.07
        T = 5
        prob = 1e-3
        M = 250
        res =  VasicekParams(r0, M,sigma, kappa, theta,T,prob=prob)
        expected=(-0.024856663930153455, 0.0087994394258510949, 22, 0.02)
        self.confirm(res, expected)
        
    def test031_DiagsSimple(self):
        'VasicekDiagonals: simple '
        sigma = 0.05
        kappa = 0.0
        theta = 0.0
        rMin = -0.02
        dr = 0.01
        N = 10
        dtau = 0.02
        res = VasicekDiagonals(sigma, kappa, theta,
                    rMin, dr, N,
                    dtau )
        expected=(array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ]), array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008,
        1.501 ,  1.5012,  1.0014]), array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ]))
        self.confirm(res, expected)
        

    def test032_Diags(self):
        'VasicekDiagonals: theta=0  '
        sigma = 0.03
        kappa = 0.5
        theta = 0.0
        rMin = -0.02
        dr = 0.01
        N = 10
        dtau = 0.02
        res = VasicekDiagonals(sigma, kappa, theta,
                    rMin, dr, N,
                    dtau)
        expected=(array([ 0.   , -0.085, -0.09 , -0.095, -0.1  , -0.105, -0.11 , -0.115,
                           -0.12 , -0.07 ]), array([ 1.0196,  1.1798,  1.18  ,  1.1802,  1.1804,  1.1806,  1.1808,
                            1.181 ,  1.1812,  1.0714]), array([-0.02 , -0.095, -0.09 , -0.085, -0.08 , -0.075, -0.07 , -0.065,
                           -0.06 ,  0.   ]))
        self.confirm(res, expected)
        

    def test035_Diags(self):
        'VasicekDiagonals: kappa theta '
        sigma = 0.03
        kappa = 0.5
        theta = 0.07
        rMin = -0.02
        dr = 0.01
        N = 10
        dtau = 0.02
        res = VasicekDiagonals(sigma, kappa, theta,
                    rMin, dr, N,
                    dtau)
        expected=(array([  0.00000000e+00,  -5.00000000e-02,  -5.50000000e-02,
            -6.00000000e-02,  -6.50000000e-02,  -7.00000000e-02,
            -7.50000000e-02,  -8.00000000e-02,  -8.50000000e-02,
             1.38777878e-17]), array([ 1.0896,  1.1798,  1.18  ,  1.1802,  1.1804,  1.1806,  1.1808,
            1.181 ,  1.1812,  1.0014]), array([-0.09 , -0.13 , -0.125, -0.12 , -0.115, -0.11 , -0.105, -0.1  ,
           -0.095,  0.   ]))
        self.confirm(res, expected)
        
    def test036_Diags(self):
        'VasicekDiagonals: uneven '
        sigma = 0.033
        kappa = 0.4554
        theta = 0.06882
        rMin = -0.0211
        dr = 0.012
        N = 13
        dtau = 0.02
        res = VasicekDiagonals(sigma, kappa, theta,
                    rMin, dr, N,
                    dtau)
        expected=(array([ 0.        , -0.04605436, -0.05060836, -0.05516236, -0.05971636,
                           -0.06427036, -0.06882436, -0.07337836, -0.07793236, -0.08248636,
                           -0.08704036, -0.09159436, -0.04104672]), array([ 1.06782728,  1.151068  ,  1.151308  ,  1.151548  ,  1.151788  ,
                            1.152028  ,  1.152268  ,  1.152508  ,  1.152748  ,  1.152988  ,
                            1.153228  ,  1.153468  ,  1.04350472]), array([-0.06824928, -0.10519564, -0.10064164, -0.09608764, -0.09153364,
                           -0.08697964, -0.08242564, -0.07787164, -0.07331764, -0.06876364,
                           -0.06420964, -0.05965564,  0.        ]))
        self.confirm(res, expected)
        
    def test041_CheckExercise(self):
        'CheckExercise: against const '
        V = array([1,0.01,100,0.5])
        eex = 1.8
        res =  CheckExercise(V,eex)
        expected=array([False,False,True, False, ], dtype=bool)
        self.confirm(res, expected)
        

    def test042_CheckExercise(self):
        'CheckExercise: against array '
        V = array([1,0.01,100,0.5])
        eex = array([0.75,0.011, 99, 0.51])
        res =  CheckExercise(V,eex)
        expected=array([ True, False,  True, False], dtype=bool)
        self.confirm(res, expected)
        

    def test051_CallExercise(self):
        'CallExercise:  simple'
        ratio = 1.0
        R = 0.02
        tau = 1
        res =  CallExercise(R,ratio,tau)
        expected=0.98019867330675525
        self.confirm(res, expected)
        
    def test052_CallExercise(self):
        'CallExercise:  '
        ratio = 1.01
        R = 0.02
        tau =3.1
        res =  CallExercise(R,ratio,tau)
        expected = 0.94928171565899977
        self.confirm(res, expected)
        
    def test060_VP(self):
        'VasicekPolicyDiagonals:  same as orig if no exer'
        from TridiagonalSolver import TridiagonalSolve
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.99561612,
                            0.99342926,  0.99124749,  0.98907061,  0.98689808,  0.98472858])
        new = array([ 1.0048125 ,  1.0024076 ,  1.00000623,  0.99760999,  0.99521953,
                            0.99283505,  0.99045655,  0.98808379,  0.98571609,  0.98335189])
        eex = 10.5 * ones( shape(new) )
        pdsub, pdd, pdsuper = VasicekPolicyDiagonals(subd,  d,  superd, old, new, eex)
        expected = (subd,  d,  superd,)
        self.confirm((pdsub, pdd, pdsuper, ), expected)
        

    def test061_VP(self):
        'VasicekPolicyDiagonals:  simple '
        from TridiagonalSolver import TridiagonalSolve
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = r_[1:11]
        new = 5*old
        eex = 35.1 *ones(shape(new))
        pdsub, pdd, pdsuper = VasicekPolicyDiagonals(subd,  d,  superd, old, new, eex)
        expected = (array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ,  0.  ,  0.  ]),
                array([ 0.9996, 1.4998, 1.5, 1.5002, 1.5004, 1.5006, 1.5008, 0.22792023,  0.25641026,  0.28490028]),
                array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ,  0.  ,  0.  ]))
        self.confirm((pdsub, pdd, pdsuper, ), expected)
        
    def test062_VP(self):
        'VasicekPolicyDiagonals:  nonconst eex '
        from TridiagonalSolver import TridiagonalSolve
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = r_[1:11]
        new = 5*old
        eex = array([3,77,3,3,3,77,3,3,77,3, ])
        pdsub, pdd, pdsuper = VasicekPolicyDiagonals(subd,  d,  superd, old, new, eex)
        expected = (array([ 0.  , -0.25,  0.  ,  0.  ,  0.  , -0.25,  0.  ,  0.  , -0.25,  0.  ]),
            array([ 0.33333333, 1.4998, 1, 1.33333333, 1.66666667, 1.5006, 2.33333333, 2.66666667, 1.5012, 3.33333333]),
            array([ 0, -0.25, 0, 0, 0, -0.25, 0, 0, -0.25, 0.  ]))
        self.confirm((pdsub, pdd, pdsuper, ), expected)
        

    def test063_VP(self):
        'VasicekPolicyDiagonals: realistic'
        from TridiagonalSolver import TridiagonalSolve
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.99561612,
                            0.99342926,  0.99124749,  0.98907061,  0.98689808,  0.98472858])
        new = array([ 1.0048125 ,  1.0024076 ,  1.00000623,  0.99760999,  0.99521953,
                            0.99283505,  0.99045655,  0.98808379,  0.98571609,  0.98335189])
        eex = 0.996* ones(shape(new))
        pdsub, pdd, pdsuper = VasicekPolicyDiagonals(subd,  d,  superd, old, new, eex)
        expected =(array([ 0.  ,  0.  ,  0.  ,  0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ]),
                array([ 1.00844436, 1.00623115, 1.00402103, 1.00181533, 1.5004, 1.5006, 1.5008, 1.501, 1.5012, 1.0014    ]),
                array([ 0.  ,  0.  ,  0.  ,  0.  , -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ]))
        self.confirm((pdsub, pdd, pdsuper, ), expected)

    def test071_Iter(self):
        'Iterate: no eex optimal'
        from TridiagonalSolver import TridiagonalSolve
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.99561612,
                            0.99342926,  0.99124749,  0.98907061,  0.98689808,  0.98472858])
        solved = array([ 1.0048125 ,  1.0024076 ,  1.00000623,  0.99760999,  0.99521953,
                            0.99283505,  0.99045655,  0.98808379,  0.98571609,  0.98335189])
        eex = 1.0996* ones( shape(old) )
        new, numPolicy = Iterate(subd,  d,  superd, old, eex, maxPolicyIterations=10)
        expected = (solved, 0)
        self.confirm((new, numPolicy, ), expected)

    def test072_Iter(self):
        'Iterate: eex optimal, no policy iter allowed'
        from TridiagonalSolver import TridiagonalSolve
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.94561612,
                            0.94342926,  0.94124749,  0.93907061,  0.93689808,  0.93472858])
        solved = array([ 0.996, 0.996, 0.996, 0.99029174,  0.95255873, 0.94412027,  0.94071176,  0.93817061,  0.93578216,  0.93342179])
        eex = 0.996* ones(shape(old))
        new, numPolicy = Iterate(subd,  d,  superd, old, eex, maxPolicyIterations=0)
        expected = (solved, 0)
        self.confirm((new, numPolicy, ), expected)


    def test073_Iter(self):
        'Iterate: eex optimal, easy to see policy iter change values (1 iter used)'
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = cast[float64](r_[1:11])
        eex = array([3,3,3,3,3,3,3,77,3,3, ])
        new, numPolicy = Iterate(subd,  d,  superd, old, eex, maxPolicyIterations=10)
        solved = array([ 1.00040016,  1.97176758,  2.82862793,  3, 3. ,3, 3, 6.32911392,  3, 3])
        expected = (solved, 1)
        self.confirm((new, numPolicy, ), expected)

    def test074_Iter(self):
        'Iterate: eex optimal, smoothed minimally by only one policy iter needed'
        subd = array([ 0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.  ])
        d = array([ 0.9996,  1.4998,  1.5   ,  1.5002,  1.5004,  1.5006,  1.5008, 1.501 ,  1.5012,  1.0014])
        superd = array([-0.  , -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,  0.  ])
        old = array([ 1.00441058,  1.00220623,  1.00000495,  0.99780807,  0.94561612,
                            0.94342926,  0.94124749,  0.93907061,  0.93689808,  0.93472858])
        solved = array([ 0.942, 0.942, 0.942, 0.942, 0.942,0.942     ,  0.9403482 ,  0.93810833,  0.93577179,  0.93342179     ])
        eex = 0.942 * ones(shape(old))
        new, numPolicy = Iterate(subd,  d,  superd, old, eex, maxPolicyIterations=10)
        expected = (solved, 1)
        self.confirm((new, numPolicy, ), expected)

    def test092_ValVector(self):
        'VasicekCallableZCBVals: no mean rev, tiny'
        r0 = 0.04
        sigma = 0.02
        kappa = 0.0
        theta = 0.07
        T = 5
        prob = 1e-2
        M = 100
        ratio = 1.0
        R = 0.02
        rs,vs = VasicekCallableZCBVals( r0, R, ratio, T,  sigma, kappa, theta, M, prob=prob )
        re,ve = (array([-0.06403744, -0.05499071, -0.04594397, -0.03689724, -0.0278505 ,
                               -0.01880377, -0.00975704, -0.0007103 ,  0.00833643,  0.01738317,
                                0.0264299 ,  0.03547663,  0.04452337,  0.0535701 ,  0.06261683,
                                0.07166357,  0.0807103 ,  0.08975704,  0.09880377,  0.1078505 ,
                                0.11689724,  0.12594397,  0.13499071,  0.14403744]),
                    array([ 0.90483742,  0.90483742,  0.90483742,  0.90483742,  0.90483742,
                              0.90483742,  0.90483742,  0.90204901,  0.8903662 ,  0.87186252,
                              0.84840112,  0.82155648,  0.79260215,  0.762533  ,  0.7320996 ,
                              0.70184521,  0.67214226,  0.64322631,  0.61522523,  0.58818137,
                              0.56206529,  0.53677973,  0.51215277,  0.48791882])  )
        self.confirm(rs,re)
        self.confirm(vs,ve)


    def test093_ValVector(self):
        'VasicekCallableZCBVals: typical values'
        r0 = 0.04
        sigma = 0.03
        kappa = 0.5
        theta = 0.07
        T = 5
        prob = 1e-6
        M = 250
        ratio = 1.0
        R = 0.02
        rs,vs = VasicekCallableZCBVals( r0, R, ratio, T,  sigma, kappa, theta, M, prob=prob )
        re,ve = (array([-0.07458404, -0.06597062, -0.05735719, -0.04874377, -0.04013035,
                               -0.03151692, -0.0229035 , -0.01429008, -0.00567665,  0.00293677,
                                0.0115502 ,  0.02016362,  0.02877704,  0.03739047,  0.04600389,
                                0.05461731,  0.06323074,  0.07184416,  0.08045759,  0.08907101,
                                0.09768443,  0.10629786,  0.11491128,  0.1235247 ,  0.13213813,
                                0.14075155,  0.14936498,  0.1579784 ,  0.16659182,  0.17520525,
                                0.18381867,  0.19243209,  0.20104552,  0.20965894]),
                    array([ 0.90483742,  0.90021179,  0.88938714,  0.8768767 ,  0.86388294,
                            0.85079493,  0.83776254,  0.8248507 ,  0.81208978,  0.79949474,
                            0.78707305,  0.77482826,  0.76276174,  0.75087357,  0.73916305,
                            0.72762894,  0.71626966,  0.7050834 ,  0.6940682 ,  0.68322199,
                            0.67254261,  0.66202787,  0.65167553,  0.64148334,  0.63144904,
                            0.62157037,  0.61184508,  0.60227091,  0.59284562,  0.58356693,
                            0.57443238,  0.56543887,  0.55658014,  0.54783418])  )
        self.confirm(rs,re)
        self.confirm(vs,ve)

    def test094_ValVector(self):
        'VasicekCallableZCBVals: typical values 2'
        r0 = 0.04
        sigma = 0.03
        kappa = 0.15
        theta = 0.07
        T = 5
        prob = 1e-6
        M = 250
        ratio = 1.0
        R = 0.02
        rs,vs = VasicekCallableZCBVals( r0, R, ratio, T,  sigma, kappa, theta, M, prob=prob )
        re,ve = (array([-0.17364915, -0.16514996, -0.15665077, -0.14815157, -0.13965238,
                       -0.13115319, -0.122654  , -0.11415481, -0.10565562, -0.09715643,
                       -0.08865724, -0.08015805, -0.07165886, -0.06315967, -0.05466048,
                       -0.04616129, -0.03766209, -0.0291629 , -0.02066371, -0.01216452,
                       -0.00366533,  0.00483386,  0.01333305,  0.02183224,  0.03033143,
                        0.03883062,  0.04732981,  0.055829  ,  0.06432819,  0.07282738,
                        0.08132658,  0.08982577,  0.09832496,  0.10682415,  0.11532334,
                        0.12382253,  0.13232172,  0.14082091,  0.1493201 ,  0.15781929,
                        0.16631848,  0.17481767,  0.18331686,  0.19181606,  0.20031525,
                        0.20881444,  0.21731363,  0.22581282,  0.23431201,  0.2428112 ,
                        0.25131039,  0.25980958,  0.26830877,  0.27680796,  0.28530715]),
                    array([ 0.90483742,  0.90483742,  0.90483742,  0.90483742,  0.90483742,
                            0.90483742,  0.90483742,  0.90483742,  0.90483742,  0.90483742,
                            0.90483742,  0.90483742,  0.90483742,  0.90483742,  0.90483742,
                            0.90483742,  0.90483742,  0.90483742,  0.89906927,  0.88860142,
                            0.87474057,  0.85847135,  0.84053248,  0.82147677,  0.80171746,
                            0.7815636 ,  0.76124646,  0.74093921,  0.72077134,  0.70083932,
                            0.68121448,  0.66194893,  0.64307989,  0.62463318,  0.60662574,
                            0.58906765,  0.57196371,  0.55531465,  0.5391181 ,  0.52336931,
                            0.50806178,  0.49318768,  0.47873823,  0.46470396,  0.45107496,
                            0.43784092,  0.42499131,  0.41251529,  0.40040153,  0.38863785,
                            0.37721019,  0.36610062,  0.35528299,  0.34471354,  0.33431036]))
        self.confirm(rs,re)
        self.confirm(vs,ve)

    def test095_ValVector(self):
        'VasicekCallableZCBVals: typical values 3'
        r0 = 0.05
        sigma = 0.03
        kappa = 0.5
        theta = 0.02
        T = 5
        prob = 1e-6
        M = 250
        ratio = 1.0
        R = 0.02
        rs,vs = VasicekCallableZCBVals( r0, R, ratio, T,  sigma, kappa, theta, M, prob=prob )
        re,ve = (array([-0.11965894, -0.11104552, -0.10243209, -0.09381867, -0.08520525,
                       -0.07659182, -0.0679784 , -0.05936498, -0.05075155, -0.04213813,
                       -0.0335247 , -0.02491128, -0.01629786, -0.00768443,  0.00092899,
                        0.00954241,  0.01815584,  0.02676926,  0.03538269,  0.04399611,
                        0.05260953,  0.06122296,  0.06983638,  0.0784498 ,  0.08706323,
                        0.09567665,  0.10429008,  0.1129035 ,  0.12151692,  0.13013035,
                        0.13874377,  0.14735719,  0.15597062,  0.16458404]),
                    array([ 0.90483742,  0.90483742,  0.90483742,  0.90483742,  0.90483742,
                            0.90483742,  0.90483742,  0.90483742,  0.90483742,  0.90483742,
                            0.90483742,  0.90483742,  0.90483742,  0.9021775 ,  0.89626926,
                            0.88834222,  0.87912294,  0.8690603 ,  0.85844283,  0.84746245,
                            0.8362508 ,  0.82490052,  0.81347832,  0.80203316,  0.79060162,
                            0.77921139,  0.76788369,  0.75663494,  0.74547793,  0.73442266,
                            0.72347683,  0.71264593,  0.70193146,  0.69131974]))
        self.confirm(rs,re)
        self.confirm(vs,ve)


    def test102_Val(self):
        'VasicekCallableZCB: no mean rev, tiny'
        r0 = 0.04
        sigma = 0.02
        kappa = 0.0
        theta = 0.07
        T = 5
        prob = 1e-2
        M = 100
        ratio = 1.0
        R = 0.02
        v = VasicekCallableZCB( r0, R, ratio, T,  sigma, kappa, theta, M, prob=prob )
        self.confirm(v,0.8070793177938063)

    def test104_Val(self):
        'VasicekCallableZCB: typical values, matches test095_ValVector'
        r0 = 0.05
        sigma = 0.03
        kappa = 0.5
        theta = 0.02
        T = 5
        prob = 1e-6
        M = 250
        ratio = 1.0
        R = 0.02
        v = VasicekCallableZCB( r0, R, ratio, T,  sigma, kappa, theta, M, prob=prob )
        self.confirm(v,0.839647498188394)



def single():
    suite = unittest.TestSuite()
    suite.addTest(TestVasicek('test074_Iter'))
    return suite

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVasicek)
    return suite
    
if __name__ == '__main__':
    tests=suite()
    #~ tests=single()
    unittest.TextTestRunner(verbosity=2).run(tests)