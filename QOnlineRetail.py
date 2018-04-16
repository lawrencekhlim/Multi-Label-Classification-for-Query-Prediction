import csv
from QLearn import QLearn

class QOnlineRetail:
    def __init__ (self):
        self.data = []
        self.training = (0, 0.5)
        self.validation = (0.5, 0.8)
        self.testing = (0.8, 1)
        
        
        
        with open ('timeseriesOnlineRetailCleaned2.csv', 'r') as f:
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
    
    
    def clean_data (self):
        data_used = []
        new_data = []
        new_header = []
        
        uncleaned_training_data = self.data[int (360*self.training[0]):int(360*self.training[1])]
        #print (len (uncleaned_training_data))
        
        sums_of_columns = [ sum(x) for x in zip(*uncleaned_training_data) ]
        #print (sums_of_columns)
        
        mean = sum (sums_of_columns) / len (sums_of_columns)
        #print ("Mean = " + str(mean))
        
        variance = 0
        for i in range (0, len (sums_of_columns)):
            variance += (sums_of_columns[i]-mean) ** 2
        variance = variance / len (sums_of_columns)
        #print ("Variance = " + str(variance))
        
        
        std_dev = variance ** 0.5
        #print ("Standard Deviation = " + str(std_dev))
    
    
        mean_of_nonzero_columns = 0
        num_of_nonzero_columns = 0
        for i in range (0, len (sums_of_columns)):
            if sums_of_columns[i] > 0:
                mean_of_nonzero_columns += sums_of_columns[i]
                num_of_nonzero_columns += 1
        mean_of_nonzero_columns = mean_of_nonzero_columns / num_of_nonzero_columns
        #print ("Mean of non-zero columns = " + str(mean_of_nonzero_columns))
        
        variance_of_nonzero_columns = 0
        for i in range (0, len (sums_of_columns)):
            if sums_of_columns[i] > 0:
                variance_of_nonzero_columns += (sums_of_columns[i]-mean_of_nonzero_columns) ** 2
        variance_of_nonzero_columns = variance_of_nonzero_columns / num_of_nonzero_columns
        #print ("Variance of non-zero columns = " + str(variance_of_nonzero_columns))


        std_dev_of_nonzero_columns = variance_of_nonzero_columns ** 0.5
        #print ("Standard Deviation of non-zero columns = " + str(std_dev_of_nonzero_columns))
        
        
        # Remove data one standard deviation below the non-zero mean
        lower_cutoff = mean_of_nonzero_columns - std_dev_of_nonzero_columns +3
        print ("Lower cutoff = " + str(lower_cutoff))
        for i in range (0, len(sums_of_columns)):
            if (sums_of_columns[i] > lower_cutoff):
                data_used.append (i)
                new_header.append (self.products[i])

        for row in range (0, len (self.data)):
            new_data.append ([])
            for col in range (0, len (self.data[row])):
                if (col in data_used):
                    new_data[row].append (self.data[row][col])
        #self.data = new_data
        #print (new_data)

        print ("Writing into CSV")
        with open ('timeseriesOnlineRetailCleaned.csv', "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow (new_header)
            writer.writerows (new_data)

        
    
    
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
    
    #test.clean_data()

    test.train_data()
    test.validate_predictor()
    #test.print_concepts()
