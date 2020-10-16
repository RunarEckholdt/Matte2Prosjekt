# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:22:52 2020

@author: Runar
"""
import numpy as np
import sympy as sy


λ = sy.Symbol('λ')

class Matrix():
    def __init__(self,array):
        self.array = np.array(array,dtype=sy.core.add.Add)
        self.shape = self.array.shape
     
    def determinant(self):
        if self.shape[0] != self.shape[1]:
            raise Exception("Matrix must be of shape AxA")
        return self.__determinant(self.array)
        
            
    def eigenValue(self):
        if self.shape[0] != self.shape[1]:
            raise Exception("Matrix must be of shape NxN to calculate eigenvalue")
        I = generateIndentityMatrix(self.shape[0])
        I = I * λ
        A = Matrix(self.array.copy())
        A = A-I
        det = A.determinant()
        det = sy.expand(det)
        print(det)
        if(type(det)!=sy.core.add.Add):
            raise Exception("No eigenValue")
        return np.array(sy.solve(det,λ))
    
    #returnerer egen verdiene og egenvektorene til matrisen i form av en array
    #første possisjon i arrayen inneholder egenverdiene og andre inneholder de tilhørende egenvektorene
    def eigenVector(self):
        eigenValues = self.eigenValue()
        I = generateIndentityMatrix(self.shape[0])
        I = I * λ
        M = Matrix(self.array.copy())
        M = M-I
        eigenVectors = self.__calcEigenVectors(eigenValues,M)
        return [eigenValues,eigenVectors]
        
    
    
    
    def gaussianElimination(self):
        a = self.array.copy()
        a = self.__echelonForm(a)
        a = self.__redusedEchelon(a)
        return Matrix(a)
    
    def transpose(self):
        a = self.array.copy()
        a = a.T
        return Matrix(a)
    
    def inverse(self):
        a = self.array.copy()
        m,n = a.shape
        tmp = Matrix(np.zeros(shape=(m,n+m),dtype=float))
        I = generateIndentityMatrix(m)
        for i in range(m):
            tmp.array[i,:] = np.append(a[i,:],I.array[i,:])
        tmp = tmp.gaussianElimination()
        output = Matrix(np.zeros(shape=(m,n),dtype=float))
        for i in range(m):
            output.array[i,:] = tmp.array[i,m:]
        return output
                    
     
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
                if(type(self.array[i][j]) == sy.core.numbers.Float or (type(self.array[i][j]) == np.float64)):
                    s = s + "%17.3f" %(self.array[i][j])
                else:
                    s = s + ("%20s" %(str(self.array[i][j])))
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
        a = np.zeros(shape=(targetShape,targetShape),dtype=sy.core.add.Add)
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
            if all(l == 0 for l in a[r:,c]): #hvis alle er 0 hopp til neste kolonne
                c += 1
                if c == n:
                    break        
            else: 
                for i in range(r,m):
                    if(type(a[i][c]) == sy.core.add.Add):
                        if(not a[i][c].is_real and sy.re(a[i][c]) == 0):
                            a[i,:] = a[i,:]*sy.I
                            for j in range(len(a[i,:])):
                                a[i][j] = sy.expand(a[i][j])
                            
                for i,v in enumerate(a[r:,c],start=r):
                    if(v == 1): #hvis en av radene har en 1'er bytt.
                        a[i,:] , a[r,:] = a[r,:].copy() , a[i,:].copy()
                        break
                if(a[r][c] == 0):
                    index = np.argmax(abs(a[r:,c])) + r  #index for største verdi i kolonne c fra rad r. + r for å få riktig index for hele kolonnen
                    a[r,:] , a[index,:] = a[index,:].copy(), a[r,:].copy()  
                elif(type(a[i][c]) == sy.core.add.Add and not a[r][c].is_real):
                    for i,v in enumerate(a[r:,c],start=r):
                        if(v.is_real):
                            a[i,:], a[r,:] = a[r,:].copy(), a[i,:].copy()
                            break
                    
                a[r,:]=a[r,:]/a[r][c]
                for i in range(r+1,m):
                    if type(a[i][c]) == sy.core.add.Add and not a[i][c].is_real:
                        if sy.re(a[i][c]) == 0: #hvis den ikke har en reel del, gang med I
                            a[i,:] = a[i,:]*sy.I
                        f = sy.re(a[i][c])
                    else:
                        f = a[i][c]
                    a[i,:] = self.__subRowAFromB(a[r,:], a[i,:],f)
                    #hvis den nå ble kvitt den reelle delen og har en imaginær del, gjør det på nytt
                    if(not a[i][c].is_real and sy.re(a[i][c]) == 0):
                       a[i,:] = a[i,:]*sy.I
                       f = a[i][c]
                       a[i,:] = self.__subRowAFromB(a[r,:], a[i,:],f)
                r+=1
                c+=1
        return a
    
    
    def __subRowAFromB(self,row1,row2,times=1):
        if(times == 0):
            return row2.copy()
        output = np.zeros(len(row1),dtype=sy.core.add.Add)
        for i in range(len(row1)):
            output[i] = row2[i] - times*row1[i]
            output[i] = sy.expand(output[i])
        return output
    
    
    def __redusedEchelon(self,a):
        m,n = a.shape
        r = 1
        c = 1
        
        while(r < m and c < n):
            if(a[r][c] == 0):
                c+=1
                
            else:
                #går fra current rad, ser oppover og gjør rad opperasjoner slik at de blir 0
                for i in range(r-1,-1,-1): 
                    f = a[i][c]
                    a[i,:] = self.__subRowAFromB(a[r,:], a[i,:], f)
                r += 1
                c += 1
        return a
    
    def __calcEigenVectors(self,eigenValues,M):
        eigenVectors = []
        for e in eigenValues:
            m = M.array.copy()
            m = Matrix(self.__subsEigenValue(m,eigenValue=e))
            m = m.gaussianElimination()
            eigenVector = EigenVector(e,m.array)
            eigenVectors.append(eigenVector)
        
        return eigenVectors
    
    def __subsEigenValue(self,M,eigenValue):
        n,m = M.shape
        for i in range(n):
            for j in range(m):
                M[i][j] = M[i][j].subs(λ,eigenValue)
        return M
    

  

  
    
    
class EigenVector():
    def __init__(self,eigenValue,a):
        self.a = a.copy()
        self.eigenValue = eigenValue
        self.symbols = sy.symbols('x y z a b c d e f g h i j k l m n o p q r s t u v w')
        a = a.copy().astype(sy.core.add.Add)
        m,n = a.shape
        
        for i in range(n):
            a[:,i] = a[:,i]*self.symbols[i]
        
        
        self.expressions = self.__extractExpressions(a)
        
    
    # def asMatrix(self):
    #     m,n = self.a.shape
    #     for i in range(n):
    #         a = self.a.copy()
    #         a[:,i] = a[:,i]*self.symbols[i]
        
        
        
    def __extractExpressions(self,a):
        
        expressions = []
        m,n = a.shape
        r = 0
        c = 0
        while(r < m and c < n):
            if(a[r][c] == 1.0*self.symbols[c] or a[r][c] == self.symbols[c]):
                if(c+1 >= n):
                    expressions.append([self.symbols[c]])
                else:
                    tmp = a[r,c+1:].copy()*-1
                    tmp = tmp[tmp!=0]
                    
                    if(len(tmp) == 0):
                        expressions.append([0])
                    else:
                        expressions.append(tmp)
                    
                c+=1
                r+=1
            elif(a[r][c] == 0):
                expressions.append([self.symbols[c]])
                c+=1
            else:
                c+=1
                r+=1
        return expressions
    
    def __str__(self):
        if(type(self.eigenValue) == sy.core.add.Add):
            s = "Eigenvalue = %10s \n" %(self.eigenValue)
        else:
            s = "Eigenvalue = %.3f \n" %(self.eigenValue)
        s += "Eigenvector = {\n"
        for i in range(len(self.expressions)):
            s+= "%s = %s\n" %(self.symbols[i],self.expressions[i])
        s+="}"
        return s
        
            
        
        


def generateIndentityMatrix(shape=2):
    d = np.ones(shape)
    i = np.diagflat(d)
    return Matrix(i)

    






    


    