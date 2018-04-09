
import scipy
import numpy as np
#Resources
#https://www.kaggle.com/carrie1/ecommerce-data/data
#https://archive.ics.uci.edu/ml/machine-learning-databases/00396/
#https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly
#https://www.youtube.com/watch?v=eHqhJylvIs4&app=desktop
#vampath

class QLearn:


    def __init__ (self):
        self.trained = False
        self.X = [] # 2d array, first dimension is the data point, second dimension is the values of the data point
        self.Y = []
    
    def train (self, X=None, Y=None):
        if X == None and Y == None:
            X = self.X
            Y = self.Y
        X1 = np.matrix (X).transpose()
        Y1 = np.matrix (Y).transpose()
        decomp = np.linalg.svd (X1)
        U = np.matrix(decomp[0])
        VT = np.matrix(decomp [2])
        V = VT.transpose()
        UT = U.transpose()




        sigma = [[0]*len(VT.getA())]*len(U.getA()[0])
        #sigma = [[0]*len(U.getA())]*len(VT.getA()[0])
        for i in range (len(decomp[1])):
            sigma [i][i] = decomp [1][i]

        sigmainv = np.matrix(np.linalg.pinv(sigma))
     
        print len(V.getA())
        print len(V.getA()[0])
        print len(sigmainv.getA())
        print len(sigmainv.getA()[0])
        print len(UT.getA())
        print len(UT.getA()[0])
        inv = V * sigmainv * UT
        inv = np.linalg.pinv(X1)
        
        self.Z = Y1 *inv
        self.trained = True

    def predict (self, input):
        if (self.trained):
            return np.matrix(self.Z * np.matrix ([input]).transpose()).transpose().getA()[0]
        else:
            return 0

    def test_model (self, input, output, verbose=True):
        total_error = 0
        for i in range (len (output)):
            prediction = self.predict (input[i]).tolist()
            err = self.l2_error (output[i], prediction)
            if verbose:
                print (str (i+1)+ ") ")
                print ("Actual:    " + str(output[i]))
                print ("Predicted: "+ str(prediction))
                print ("L2 Error:  " + str (err))
                #print ("Std Dev:   " + str (err ** (0.5)))
                print ("")
            total_error += err
        total_error /= len (output)
        print ("Average Error: " + str (total_error))

    def l2_error (self, real, prediction):
        error = 0
        for i in range (len (real)):
            error += (real[i] - prediction[i]) ** 2
        return error


    def add_data_point (self, x, y):
        self.X.append (x)
        self.Y.append (y)

    def set_training_data (self, input, output):
        self.X = input
        self.Y = output


