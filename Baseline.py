import numpy as np

class Baseline:
    def train (self, X, Y):
        Y1 = np.matrix (Y).transpose().tolist()
        y1_rows = len (Y1)
        y1_cols = len (Y1[0])
        self.output_size = y1_rows
    
    def predict (self, input):
        out = self.output_size * [0]
        for i in range (0, self.output_size):
            out[i] = input [len (input)- self.output_size+i]
        return out


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
                print (len (output[i]))
        total_error = total_error / len (output)
        print ("Average Error L2: " + str (total_error))
        
        rmsd = (total_error / len (output[0]))**0.5
        print ("RMSD: " + str(rmsd))

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
