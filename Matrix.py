# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:22:52 2020

@author: Runar
"""
import numpy as np
import sympy as sy

class Matrix():
    def __init__(self,array):
        self.array = np.array(array,dtype=sy.core.add.Add)
        self.shape = self.array.shape
     
    def determinant(self):
        if self.shape[0] != self.shape[1]:
            raise Exception("Matrix must be of shape AxA")
        return self.__determinant(self.array)
        
            
    
    def gaussianElimination(self):
        a = self.array.copy()
        a = self.__echelonForm(a)
        a = self.__redusedEchelon(a)
        return Matrix(a)
    
    def transpose(self):
        a = self.array.copy()
        a = a.T
        return Matrix(a)
                    
     
    #overlastning av + operatoren
    def __add__(self,addend2):
        if type(addend2) == Matrix:
            addend2 = addend2.array
        elif type(addend2) != np.ndarray:
            raise Exception("Unsupported datatype for the '+' operator on <Class Matrix>")
        addend1 = self.array
        
        if addend1.shape != addend2.shape:
            raise Exception("Matrix must be of same shape")
        output = np.zeros(shape=self.shape,dtype=float)
        #går igjennom hver verdi i begge arrayene og legger dem sammen
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                output[i][j]=addend1[i][j]+addend2[i][j]
        return Matrix(output)
    
    #overlasting av str() funksjonen for printing
    def __str__(self):
        s = ""
        for i in range(self.shape[0]):
            
            for j in range(self.shape[1]):
                s = s + str(self.array[i][j])
                s = s + "             "
            s = s + "\n"
        return s
    
    
    
    
    #overlastning av * operatoren
    def __mul__(self,factor2):
        factor1 = self.array
        if type(factor2) == int or type(factor2) == float or type(factor2) == np.number or type(factor2) == sy.core.symbol.Symbol:
            output = np.zeros(shape = factor1.shape,dtype=sy.core.add.Add)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    output[i][j] = factor1[i][j] * factor2
            return Matrix(output)
        
        
        if type(factor2) == Matrix:
            factor2 = factor2.array
        elif type(factor2) == list:
            factor2 = np.array(factor2)
        elif type(factor2) != np.ndarray:
            raise Exception("Unsupported datatype for the '*' operator on <Class Matrix>")
            
        if factor1.shape[1] != factor2.shape[0]:
            raise Exception("Matrixes must be of shape AxB * BxC")
        outputShape = (factor1.shape[0],factor2.shape[1])
        output = np.zeros(shape=outputShape,dtype=np.float64)
        #for hver rad i matrise 1
        for i in range(outputShape[0]):
            #for hver rad i matrise 2
            for j in range(outputShape[1]):
                output[i][j] = self.__rowMulColumn(factor1[i,:],factor2[:,j])
        return Matrix(output)
    
    #overlastning av - operatoren
    def __sub__(self,subtrahend):
        minuend = self.array
        if type(subtrahend) == Matrix:
            subtrahend = subtrahend.array
        elif type(subtrahend!=np.ndarray):
            raise Exception("Unsupported datatype for '-' operator on <Class Matrix>")
        if minuend.shape != subtrahend.shape:
            raise Exception("Matrix must be of same shape")
        difference = np.zeros(shape=self.shape,dtype=sy.core.add.Add)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                difference[i][j]=minuend[i][j]-subtrahend[i][j]
        return Matrix(difference)
    
    def __rowMulColumn(self,row,column):
        s = 0
        #summerer produktet av hvert element per l
        for i in range(len(row)):
            s+= row[i] * column[i]
        return s
    
    
    
    #regner seg rekursivt nedover til arrayen er en 2x2 matrise. Da regner den 2. determinanten
    def __determinant(self,array):
        shape = array.shape[0]
        if shape == 2:
            return self.__secondDeterminant(array)
        nextShape = shape-1
        s = 0
        for i in range(shape):
            a = self.__fetchNextDet(array,i,nextShape)
            # +x*|a1|-y*|a2|+z*|a3|
            s+= (-1)**i*array[0][i]*self.__determinant(a) 
        return s
    
    
    
    def __fetchNextDet(self,array,i,targetShape):
        a = np.zeros(shape=(targetShape,targetShape))
        f = 0
        #lager et nytt array som tar alle rader med unntak av rad i og tar ikke med første rad
        for j in range(targetShape):
            if f == i:
                f+= 1
            a[:,j] = array[1:,f] #kolonne f men ikke 0. element
            f += 1
        return a
    
    def __secondDeterminant(self,arr):
        a,b = arr[0,:] #rad 1
        c,d = arr[1,:] #rad 2
        return a*d-b*c


    def __echelonForm(self, a):
        m,n = a.shape
        r = 0
        c = 0
        while(r < m and c < n):
            for i in range(r,m):
                if(a[i][c] == 1): #hvis en av radene har en 1'er bytt.
                    a[i,:] , a[r,:] = a[r,:] , a[i,:]
                    break
            if all(l == 0 for l in a[:,c]): #hvis alle er 0 hopp til neste kolonne
                c += 1
            else:
                a[r,:]=a[r,:]/a[r][c]
                for i in range(r+1,m):
                    f = a[i][c]
                    a[i,:] = self.__subRowAFromB(a[r,:], a[i,:],f)
                r+=1
                c+=1
        return a
    def __subRowAFromB(self,row1,row2,times=1):
        output = np.zeros(len(row1))
        for i in range(len(row1)):
            output[i] = row2[i] - times*row1[i]
        return output
    def __redusedEchelon(self,a):
        m,n = a.shape
        r = 1
        c = 1
        while(r < m and c < n):
            for i in range(r-1,-1,-1):
                f = a[i][c]
                a[i,:] = self.__subRowAFromB(a[r,:], a[i,:], f)
            r += 1
            c += 1
        return a
    
    

def generateIndentityMatrix(shape=2):
    d = np.ones(shape)
    i = np.diagflat(d)
    return Matrix(i)

    

A = Matrix([[3.0,7.0,3.0,2.0,3.0,4.0],
            [4.0,2.0,9.0,2.0,3.0,2.0],
            [7.0,0.0,1.0,2.0,3.0,1.0],
            [2.0,2.0,2.0,2.0,3.0,3.0],
            [3.0,3.0,3.0,3.0,3.0,2.0]])
B = Matrix([[0,1,0,2,2],
            [1,0,1,2,2],
            [0,1,0,4,2]])


C = Matrix([[1,0,1],
            [0,1,0],
            [1,0,1]])

F = Matrix([[1,0,0],
            [0,1,0],
            [0,0,1],
            [0,0,0],
            [0,0,0]])


λ = sy.Symbol('λ')




H = generateIndentityMatrix(3)

D = H * C
print(D)
H = H * λ
print(H)

G = C - H

print(G)

#detA = A.determinant()


# print(C)
#print(D)
# print(E)
#print(detA)

#F = A.gaussianElimination()


    


    