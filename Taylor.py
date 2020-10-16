# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:32:56 2020

@author: Runar
"""
from sympy.utilities.lambdify import lambdify
import sympy as sy
import numpy as np
from math import factorial
import matplotlib.pyplot as plt

a,x = sy.symbols('a x')


def mauclaurin(expr,terms=5):
    if(type(expr) != sy.core.add.Add):
        expr = sy.sympify(expr)
    expression = expr
    for i in range(1,terms):
        tmp = sy.diff(expr,x,i)
        tmp = (tmp*a**i)/factorial(i)
        expression = expression + tmp
    expression = expression.subs(x,0)
    expression = expression.subs(a,x)
    return expression



def plotExpr(expr,start=0,end=10,resolution=1000):
    xVals = np.linspace(start,end,resolution)
    yVals = np.zeros(len(xVals))
    f = lambdify(x,expr)
    for i,xVal in enumerate(xVals):
        yVals[i] = f(xVal)
    plt.plot(xVals,yVals)
        
    


