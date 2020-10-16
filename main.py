# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:19:17 2020

@author: Runar
"""

import Matrix



a = Matrix.Matrix([[1,-2,3],
                   [1,0,1],
                   [1,3,-2]])

vecs = a.eigenVector()

for vec in vecs[1]:
    print(vec)
    
    