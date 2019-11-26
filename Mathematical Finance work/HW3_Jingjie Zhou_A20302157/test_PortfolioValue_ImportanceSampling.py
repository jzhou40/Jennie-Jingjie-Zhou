from __future__ import division
import string, unittest, datetime
import copy as pycopy
import numpy as np
import numpy.linalg
from numpy.random import rand,randn,seed
from pprint import pprint
from PortfolioValue import importanceSampleDetails

from contextlib import contextmanager

import scipy.stats 

np.set_printoptions(precision=15)

"""
Assuming a function with the following behavior:
def importanceSampleDetails(originalUniformSamples, correlation_in,shifts_in=0):
    
    Gaussian importance sampling:  take the multivariate random normal
     and shift it by the given amount.  Return the Radon-Nikodym derivative
     and corresponding uniform values as numpy array objects in a dict with
     keys 'Weights' and 'Samples'
    
run unit tests
"""

class TestPortfolioImportance(unittest.TestCase):
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


    def test_071a_ImportanceDetails1Sample(self):
        ''' Importance sampling, single sample (a)'''
        cor_mx = [[1, 0.2, 0.3], [0.2, 1, 0.4], [0.3, 0.4, 1]]
        # Get uniform samples such that unshifted correlated values equal zbase
        zbase = np.array([0.5, 0.5, 0.5])
        ch = numpy.linalg.cholesky(cor_mx)
        zcor = numpy.linalg.solve(ch,zbase)
        u = scipy.stats.distributions.norm.cdf(zcor)
        print zcor.shape
        print u.shape
        # Make shift and run
        s = np.array([1.5, 1.5, 1.5]) # Pretty big shift!
        res = importanceSampleDetails(u, cor_mx, s )
        expected =  {'Weights': 0.029092667011632455,
                        'Samples': np.array([ 0.97724987,  0.97724987,  0.97724987])}
        self.confirm(res,expected)

    def test_071b_ImportanceDetails1Sample(self):
        ''' Importance sampling, single sample (b)'''
        cor_mx = [[1, 0.2, 0.3], [0.2, 1, 0.4], [0.3, 0.4, 1]]
        # Get uniform samples such that unshifted correlated values equal zbase
        zbase = np.array([0.15, 0.15, 0.15])
        ch = numpy.linalg.cholesky(cor_mx)
        zcor = numpy.linalg.solve(ch,zbase)
        u = scipy.stats.distributions.norm.cdf(zcor)
        # Make shift and run
        s = np.array([1.5, 1.5, 1.5]) # Pretty big
        res = importanceSampleDetails(u, cor_mx, s )
        expected =  {'Weights': 0.078329282070922004,
                        'Samples': np.array([ 0.95052853,  0.95052853,  0.95052853])}
        self.confirm(res,expected)

    def test_072_ImportanceDetails2Sample(self):
        ''' Importance sampling, 2 sample '''
        def chinv(x):
            return numpy.linalg.solve(ch,x)
        cor_mx = [[1, 0.2, 0.3], [0.2, 1, 0.4], [0.3, 0.4, 1]]
        # Get uniform samples such that unshifted correlated values equal zbase
        zbase = np.array([[0.5, 0.5, 0.5],[0.15, 0.15, 0.15], ] )
        ch = numpy.linalg.cholesky(cor_mx)
        zcor = np.apply_along_axis(chinv, 1, zbase)
        u = scipy.stats.distributions.norm.cdf( zcor )
        # Make shift and run
        s = np.array([1.5, 1.5, 1.5]) # Pretty big
        res = importanceSampleDetails(u, cor_mx, s )
        expected =  {'Weights': np.array([ 0.02909267,  0.07832928]),
                    'Samples': np.array([[ 0.97724987,  0.97724987,  0.97724987],
                                        [ 0.95052853,  0.95052853,  0.95052853]])}
        self.confirm(res,expected)

    def test_073_ImportanceDetails2SampleConstCorr(self):
        ''' Importance sampling, 2 sample '''
        def chinv(x):
            return numpy.linalg.solve(ch,x)
        cor_mx = [[1, 0.2, 0.3], [0.2, 1, 0.4], [0.3, 0.4, 1]]
        cor = 0.25
        # Get uniform samples such that unshifted correlated values equal zbase
        zbase = np.array([[0.5, 0.5, 0.5],[0.15, 0.15, 0.15], ] )
        ch = numpy.linalg.cholesky(cor_mx)
        zcor = np.apply_along_axis(chinv, 1, zbase)
        u = scipy.stats.distributions.norm.cdf( zcor )
        # Make shift and run
        s = np.array([.5, .5, .5]) 
        res = importanceSampleDetails(u, cor, s ) # note const correl
        expected =  {'Weights': np.array([ 0.480823456169431,  0.673897987506907]),
                    'Samples': np.array([[ 0.841344746068543,  0.846203273074542,  0.822901869580183],
                                        [ 0.742153889194135,  0.744115415048617,  0.734979748711457]])}
        self.confirm(res,expected)



def single():
    suite = unittest.TestSuite()
    suite.addTest(TestPortfolioRisk('test_073_ImportanceDetails2SampleConstCorr'))
    return suite

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPortfolioImportance)
    return suite
    
if __name__ == '__main__':
    tests=suite()
    #tests=single()
    unittest.TextTestRunner(verbosity=2).run(tests)