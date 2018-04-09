import scipy
import numpy as np


def createSingularValuesMatrix (U, sigma, VT):
    numCols = len(VT[0])   # number of columns is equal to the number of rows of VT
    numRows = len(U)       # number of rows is equal to the number of columns of U
    
    arr = [[0 for col in range (numCols)] for row in range (numRows)]
    for i in range (len(sigma)):
        arr[i][i] = sigma[i]
    return arr

def find_inverse (U, sigma, VT):
    # Finding the pseudoinverse using the singular value decomposition
    # The pseudoinverse using an SVD is equal to
    # V * sigma^-1 * UT
    V = np.matrix(VT).transpose().getA()
    UT = np.matrix(U).transpose().getA()
    numCols = len (UT[0])      # number of columns is equal to the number of rows of UT
    numRows = len (V)       # number of rows is equal to the number of columns of V
    
    arr = [[0 for col in range (numCols)] for row in range (numRows)]
    for i in range (len(sigma)):
        if (sigma[i] != 0): # Avoid Divide by zero
            arr[i][i] = 1/sigma[i]
        #arr [i][i] = 1/sigma[i]
    # returns V, sigma^-1, UT
    return (V, arr, UT)

# Try dimensionality reduction
def dimension_reduction (U, sigma, VT):
    ktruncate = 0
    while (ktruncate < len (sigma) and sigma[ktruncate] > 0.001):
        ktruncate+= 1
#Utruncate = [



arr = [[1, 1, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 0]]

decomp = np.linalg.svd (arr)
U = decomp [0]
#U = np.matrix(decomp [0])
singularValues = decomp[1]
#singularValues = np.matrix(decomp [1])
VT = decomp [2]
#V = np.matrix(decomp [2])
#print (decomp)

print ("U = ")
print (U)
print ("singularValues = ")
print (singularValues)
print ("VT = ")
print (VT)

print ("Sigma = ")
sigma = createSingularValuesMatrix(U, singularValues, VT)
print (sigma)

print ("U * Sigma * VT = ")
prod = np.matmul(np.matmul(U, sigma), VT)
print (prod)

print ()
print ()
print ("Finding the inverse:")
inv = find_inverse (U, singularValues, VT)
#print (inv)
completeInverse = np.matmul(np.matmul(inv[0], inv[1]), inv[2])
print (completeInverse)

print ("Actual Pseudoinverse:")
pinv = np.linalg.pinv(arr)
print (pinv)

