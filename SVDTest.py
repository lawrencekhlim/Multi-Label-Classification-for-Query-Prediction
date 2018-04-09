import scipy
import numpy as np


def createSingularValuesMatrix (U, sigma, VT):
    numCols = len(VT)   # number of columns is equal to the number of rows of VT
    numRows = len(U[0])       # number of rows is equal to the number of columns of U
    
    arr = [[0 for col in range (numCols)] for row in range (numRows)]
    
    smaller = numCols
    if numRows < smaller:
        smaller = numRows
    for i in range (smaller):
        arr[i][i] = sigma[i]
    return arr

def find_inverse (U, sigma, VT):
    # Finding the pseudoinverse using the singular value decomposition
    # The pseudoinverse using an SVD is equal to
    # V * sigma^-1 * UT
    V = np.matrix(VT).transpose().getA()
    UT = np.matrix(U).transpose().getA()
    numCols = len (UT)      # number of columns is equal to the number of rows of UT
    numRows = len (V[0])       # number of rows is equal to the number of columns of V
    
    arr = [[0 for col in range (numCols)] for row in range (numRows)]
    
    smaller = numCols
    if numRows < smaller:
        smaller = numRows
    for i in range (smaller):
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
    # U will become an (m by r) size matrix (m rows and r columns)
    # sigma will become an (r by r) size matrix (r rows and r columns)
    # VT will become a (r by n) size matrix (r rows and n columns)

    Utruncated = [[U[row][col] for col in range (ktruncate)] for row in range (len(U))]

    VTtruncated = [[VT[row][col] for col in range (len (VT[row]))] for row in range (ktruncate)]

    '''
    arr = [[0 for col in range (ktruncate)] for row in range (ktruncate)]
    for i in range (len(ktruncate)):
        arr[i][i] = sigma[i]
    '''
    return (Utruncated, sigma, VTtruncated)

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

print ()
print ()

print ("Dimensionality Reduction")
trunc = dimension_reduction (U, singularValues, VT)
truncsigma = createSingularValuesMatrix (trunc[0], singularValues, trunc[2])

print ("U = ")
print (trunc[0])
print ("truncated sigma = ")
print (truncsigma)
print ("VT = ")
print (trunc[2])

print ("U * Sigma * VT = ")
prod = np.matmul(np.matmul(trunc[0], truncsigma), trunc[2])
print (prod)

print ()
print ()
print ("Finding the inverse:")
inv2 = find_inverse (trunc[0], singularValues, trunc[2])
print (inv2)
completeInverse2 = np.matmul(np.matmul(inv2[0], inv2[1]), inv2[2])
print (completeInverse2)

print ("Actual Pseudoinverse:")
pinv = np.linalg.pinv(arr)
print (pinv)


