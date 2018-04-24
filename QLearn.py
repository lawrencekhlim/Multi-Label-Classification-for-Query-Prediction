
import scipy
import numpy as np
#Resources
#https://www.kaggle.com/carrie1/ecommerce-data/data
#https://archive.ics.uci.edu/ml/machine-learning-databases/00396/
#https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly
#https://www.youtube.com/watch?v=eHqhJylvIs4&app=desktop

# PhD. Vaibhav
# Victor

# TODO: Regularization to prevent overfitting
# Synthesis of another dataset?
# Make sure that the math absolutely checks out?
# Understand the meaning of "concepts" in SVD
# Provide better print statements for a better understanding of data


# Do a write up with references to the code
# Remove bias from the dataset; in particular, omit if 2 standard deviations below the mean
# Get a quantifiable answer to Bias

class QLearn:


    def __init__ (self, threshold=0.5):
        self.trained = False
        self.X = [] # 2d array, first dimension is the data point, second dimension is the values of the data point
        self.Y = []
        self.threshold = threshold
    
    def train (self, X=None, Y=None):
        if X == None and Y == None:
            X = self.X
            Y = self.Y
        
        X1 = np.matrix (X).transpose().tolist()
        Y1 = np.matrix (Y).transpose().tolist()
        
        
        #decomp = np.linalg.svd (X1)
        #U = decomp[0]
        #singularValues = decomp [1]
        #VT = decomp [2]
        
        # Do dimension reduction
        #self.truncated = self.dimension_reduction (U, singularValues, VT)
        
        # Computer the inverse
        #SVD_inverse = self.find_inverse (self.truncated[0], self.truncated[1], self.truncated[2])
        #inv = np.matmul(np.matmul(SVD_inverse[0], SVD_inverse[1]), SVD_inverse[2])
        
        # Reliable and computationally faster than
        inv = np.linalg.pinv(X1)
        
        self.Z = np.matmul(Y1, inv)
        self.trained = True
            
            
    def createSingularValuesMatrix (self, U, sigma, VT):
        numCols = len(VT)   # number of columns is equal to the number of rows of VT
        numRows = len(U[0])       # number of rows is equal to the number of columns of U

        arr = [[0 for col in range (numCols)] for row in range (numRows)]
    
        smaller = numCols
        if numRows < smaller:
            smaller = numRows
        for i in range (smaller):
            arr[i][i] = sigma[i]
        return arr

    def find_inverse (self, U, sigma, VT):
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
    def dimension_reduction (self, U, sigma, VT):
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

    def predict (self, input, regularize=True):
        if (self.trained):
            vec = np.matrix(self.Z * np.matrix ([input]).transpose()).transpose().getA()[0].tolist()
            if (regularize):
                return self.regularize (vec, self.threshold)
            else:
                return vec
        else:
            return 0

    def test_model (self, input, output, verbose=True):
        total_error = 0
        predictions = []
        
        for i in range (len (output)):
            prediction = self.predict (input[i])
            predictions.append (prediction)
            err = self.l2_error (output[i], prediction)
            total_error += err
            if verbose:
                print (str (i+1)+ ") ")
                print ("\tActual\t\tPredicted")
                for prod in range (0, len (output[i])):
                    print ("\t"+str(output [i][prod])+"\t\t"+str(prediction[prod]))
                #print ("Actual:    " + str(output[i]))
                #print ("Predicted: "+ str(prediction))
                print ("L2 Error:  " + str (err))
                print ("Std Dev:   " + str ((err/len (output[i])) ** (0.5)))
                print ("")
                #print (len (output[i]))
        total_error = total_error / len (output)
        print ("Average Error L2: " + str (total_error))
        
        rmsd = (total_error / len (output[0]))**0.5
        print ("RMSD: " + str(rmsd))
        
        #coeffs = self.coeff_of_determination (output, predictions)
        #print ("Coefficients of determination: " + str (coeffs))
        
    
    def print_concepts (self, number_of_concepts=1):
        
        """
        rank = len (self.truncated[2]) # The rank is equal to the number of rows in V transpose
        print ("Rank = " + str(rank))
        
        num = number_of_concepts
        if (rank < num):
            num = rank
        
        for i in range (num):
            Ucol = np.matrix([row[i] for row in self.truncated[0]]).transpose()
            sigma = self.truncated [1][i]
            VTrow = self.truncated [2][i]
            concept = np.matrix(Ucol) * np.matrix (VTrow)
            concept = np.array(concept) * sigma
            
            print ("Concept " + str (i) + ") with sigma = " + str (sigma))
            print (concept.tolist())
            print ("")
            print ("")
        """
    
    """
        vector: a list of values (typically between 0 and 1)
        threshold: The threshold in which the values must be above to be 0 or 1
        
        Turns values in the list either 1 if it that value is greater than the threshold or
        0 if smaller than the threshold value.
    """
    def regularize (self, vector, threshold):
        for i in range (len (vector)):
            if (vector[i] >= threshold):
                vector[i] = 1
            else:
                vector[i] = 0
        return vector
    

    def l2_error (self, real, prediction):
        error = 0
        for i in range (len (real)):
            error += (real[i] - prediction[i]) ** 2
        return error
    
    def mean (self, list):
        return sum (list) / len (list)
    
    def variance (self, list):
        mean = self.mean (list)
        variance = 0
        for value in list:
            variance += (value - mean) ** 2
        return variance / len(list)
  
  
  
    def coeff_of_determination (self, real, prediction):

        #sums_of_columns_real = [ sum(x) for x in zip(*real) ]
        #sums_of_columns_prediction = [ sum(x) for x in zip(*prediction) ]
        #print (prediction)
        
        r_squared = []
        for i in range (0, len (real[0])):
            col_real = [row[i] for row in real]
            col_prediction = [row[i] for row in prediction]

            mean = self.mean(col_real)
            #ss_total = self.variance (col_real) * len (col_real)
            
            ss_total = 0
            ss_reg = 0
            for testnum in range (0, len(col_prediction)):
                ss_total += (col_real[testnum] - mean) ** 2
                ss_reg += (col_prediction[testnum] - mean) ** 2

            if (ss_total == 0):
                r_squared.append (-1)
            else:
                r_squared.append (ss_reg/ss_total)
        return r_squared


    """
            Deprecated
    """
    """
    def add_data_point (self, x, y):
        self.X.append (x)
        self.Y.append (y)
    """


    def set_training_data (self, input, output):
        self.X = input
        self.Y = output


