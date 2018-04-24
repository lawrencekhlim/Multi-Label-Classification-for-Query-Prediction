
from Model import NaiveModel
import numpy as numpy

class AverageModel(NaiveModel):
    def predict (self, input):
        out = self.output_size * [0]
        window_size = int(len (input) / self.output_size)
        
        for i in range (self.output_size):
            for b in range (window_size):
                out [i] = out[i] + input [b * window_size + i]
            out[i] = out[i] / window_size
        return out

