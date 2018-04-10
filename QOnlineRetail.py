import csv
from QLearn import QLearn

class QOnlineRetail:
    def __init__ (self):
        self.data = []
        self.training = (0, 0.5)
        self.validation = (0.5, 0.8)
        self.testing = (0.8, 1)
        
        
        with open ('timeseriesOnlineRetail.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            title_row = True
            for row in reader:
                if title_row:
                    self.products = row
                    title_row = False
                else:
                    integer_data = [int(i) for i in row]
                    self.data.append (integer_data)
        #print (self.data)
        self.predictor = QLearn()
    
    
    def validate_predictor (self):
        input = []
        output = []
        
        week_data = []
        for days in range (int (360*self.validation[0]), int (360*self.validation[0])+7):
            today = self.data[days]
            week_data = week_data+ today
    
    
        for days in range (int (360*self.validation[0])+7, int(360*self.validation[1])):
            today = self.data[days]
            input.append (week_data)
            output.append (today)
            week_data = week_data+ today
            for i in range (len (self.data[0])):
                week_data.pop (0)
        self.predictor.test_model (input, output)
    
    def train_data (self):
        input = []
        output = []
        
        week_data = []
        for days in range (int (360*self.training[0]), 7):
            today = self.data[days]
            week_data = week_data+ today
        
        
        for days in range (7, int(360*self.training[1])):
            today = self.data[days]
            input.append (week_data)
            output.append (today)
            week_data = week_data+ today
            for i in range (len (self.data[0])):
                week_data.pop (0)
        self.predictor.set_training_data (input, output)
        print ("Training Model...")
        self.predictor.train()
        print ("... Done Training")


    def print_concepts (self):
        self.predictor.print_concepts()


if __name__== "__main__":
    test = QOnlineRetail ()
    test.train_data()
    test.validate_predictor()
    test.print_concepts()
