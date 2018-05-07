import numpy as np
import keras.layers as L
import keras.models as M

import numpy


#https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features
import numpy as np

class RNNModel:
    def __init__ (self, threshold=0.5, regularization=False):
        self.threshold = threshold
        self.regularization = regularization

    
    def train (self, X, Y):
        """
        Y1 = np.matrix (Y).transpose().tolist()
        y1_rows = len (Y1)
        y1_cols = len (Y1[0])
        self.input_output_size = y1_rows
        X1 = np.matrix (X).transpose().tolist()
        x1_rows = len (X1)
        x1_cols = len (X1[0])
        """
        
        self.input_output_size = len (Y[0])
        self.window_size = int(len (X[0])/len (Y[0]))
        print ("self.window_size = ", self.window_size)
        print ("self.input_output_size = ", self.input_output_size)
        data_points = len (Y)
        
        #print (self.window_size)
        #print (self.input_output_size)
        #print (data_points)
        
        data_x = []
        data_y = []
        for a in range (0, data_points):
            #data_point_y = []
            #data_point_y.append (np.array(Y[a]))
            #data_y.append (np.array(data_point_y))
            data_y.append(Y[a])
            
            data_point_x = []
            for b in range (0, self.window_size):
                data_point_x.append (X[a][self.input_output_size*b:self.input_output_size*(b+1)])
                #print ("Length ", len (data_point_x[b]))
            data_x.append (np.array(data_point_x))
        data_x_np = np.array(data_x)
        data_y_np = np.array(data_y)
        
        #print (len(data_x_np))
        #print (len(data_x_np[0]))
        #print (len(data_x_np[0][0]))
        #print (len(data_y_np))
        #print (len(data_y_np[0]))
        #print (len(data_y_np[0][0]))

        print ("Completed array compiling")
        
        print ("Creating input")
        #self.window_size is the number of time steps
        #self.input_output_size is the number of features
        #This sets up the model so it can support the chosen time step size and number of features
        model_input = L.Input (shape=(self.window_size, self.input_output_size))
        
        print ("Creating output")
        model_output = L.LSTM (self.input_output_size, activation='softmax') (model_input)
        
        print ("Creating model")
        self.model = M.Model (input=model_input, output=model_output)
            
        print ("Compiling model")
        self.model.compile (loss='mean_absolute_error', optimizer='sgd')
        
        
        print ("Fitting data")
        self.model.fit(data_x_np, data_y_np, epochs=5, batch_size=32)
        
        
    """
    def predict (self, input):
        out = self.output_size * [0]
        for i in range (0, self.output_size):
            out[i] = input [len (input)- self.output_size+i]
        return out
    """
    def test_model_keras (self, X, Y):
        """
        Y1 = np.matrix (Y).transpose().tolist()
        y1_rows = len (Y1)
        y1_cols = len (Y1[0])
        
        X1 = np.matrix (X).transpose().tolist()
        x1_rows = len (X1)
        x1_cols = len (X1[0])
        """
        
        data_points = len (Y)
        
        #data_points = y1_cols
        
        data_x = []
        data_y = []
        for a in range (0, data_points):
            #data_point_y = []
            #data_point_y.append (Y[a])
            #data_y.append (np.array(data_point_y))
            data_y.append(Y[a])
            
            data_point_x = []
            for b in range (0, self.window_size):
                data_point_x.append (np.array(X[a][self.input_output_size*b:self.input_output_size*(b+1)]))
            data_x.append (np.array(data_point_x))
        data_x_np = np.array(data_x)
        data_y_np = np.array(data_y)
        
        #print (len(data_x_np))
        #print (len(data_x_np[0]))
        #print (len(data_x_np[0][0]))
        #print (len(data_y_np))
        #print (len(data_y_np[0]))

        
    
        loss_and_metrics = self.model.evaluate(data_x_np, data_y_np, batch_size=128)
        print (loss_and_metrics)
    
    
    def test_model (self, input, output, verbose=True):
        total_l2_error = 0
        total_l1_error = 0
        predictions = []
        
        for i in range (len (output)):
            data_x = []
            data_point_x = []
            for b in range (0, self.window_size):
                data_point_x.append (np.array(input[i][self.input_output_size*b:self.input_output_size*(b+1)]))
            data_x = np.array([data_point_x])
            prediction = self.model.predict (data_x)[0]
            predictions.append (prediction)
            err = self.l2_error (output[i], prediction)
            total_l2_error += err
            err = self.l1_error (output[i], prediction)
            total_l1_error += err
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
        total_l2_error = total_l2_error / len (output)
        print ("Average Error L2: " + str (total_l2_error))
        
        rmsd = (total_l2_error / len (output[0]))**0.5
        print ("RMSD: " + str(rmsd))
        
        total_l1_error = total_l1_error / len (output)
        print ("Average Error L1: " + str (total_l2_error))
        
        average_deviation = (total_l1_error / len (output[0]))
        print ("Average Deviation per Query: " + str(average_deviation))
        
        #coeffs = self.coeff_of_determination (output, predictions)
        #print ("Coefficients of determination: " + str (coeffs))
        
        cntng_table = self.contingency_table (output, predictions)
        print (cntng_table[0])
    
    def l1_error (self, real, prediction):
        error = 0
        for i in range (len (real)):
            error += abs(real[i] - prediction[i])
        return error
    
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
    
    """
        Returns a contingency table
        """
    def contingency_table (self, real, prediction):
        table = []
        for i in range (0, len (real[0])):
            table.append ([0, 0, 0, 0])
            for b in range (0, len (real)):
                if (real[b][i] == 1):
                    if (prediction [b][i] >= self.threshold):
                        table[i][0] += 1 # True positive
                    else:
                        table[i][2] += 1 # False negative
                else:
                    if (prediction[b][i] >= self.threshold):
                        table[i][1] += 1 # False positive
                    else:
                        table[i][3] += 1 # True negative
        return table
