__author__ = 'student'
from BS import bs
import unittest
import os, traceback
from copy import copy
from numpy import array, sqrt, nan, isnan

class  testbs(unittest.TestCase):
    def confirm( self, result, expected, digits=3, logOnErr=True):
        res = copy( result )
        try:
            self.confirm_iter( res, expected, digits=digits)
        except:
            if logOnErr:
                print('Did Not Match Expected:\n\t\t%r'%(result,))
            raise

    def confirm_iter( self, result, expected, digits=3, nanEquivToNone=True):
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
            if nanEquivToNone and expected in (None, nan):
                    self.assertTrue( result is None or isnan(result) )
            else:
                    self.assertAlmostEqual(result, expected, digits)

    def test01(self):
        expected = (1.9963, 0.1813, 7.7326)
        result = bs("call",35.0,25.0,0.03,1.5,0.5,0.5)
        self.confirm(result[:],expected[:])

    def test02(self):
        expected =  (0.2071, 0.0328, 2.0272)
        result = bs("call",25.0,15.0,0.05,2.5,0.4,0.6)
        self.confirm(result[:],expected[:])
if __name__ == '__main__':
    #~ unittest.TextTestRunner(verbosity=2).run(single())
    unittest.main()
