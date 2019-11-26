#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:36:23 2019

@author: chengzhao
"""

from scipy.special import comb
total = 0
for i in range(13,26):
    total += comb(25,i)*(0.6**i)*(0.4**(25-i))
print(total)
total1 = 0
for i in range(13,26):
    total1 += comb(25,i)*(0.45**i)*(0.55**(25-i))
print(total1)