from Query import Query
from ChainQuery import ChainQuery
from GroupQuery import GroupQuery
from CyclicalQuery import CyclicalQuery
from QLearn import QLearn

class QTester:
    def __init__ (self):
        self.queries = [None] * 7
    
        self.queries[0] = ChainQuery (0.15, 0.9)
        self.queries[1] = GroupQuery (0.1, self.queries[0], 0.95)
        self.queries[2] = ChainQuery (0.15, 0.92)
        self.queries[3] = GroupQuery (0.95, self.queries[2], 0.1)
        self.queries[4] = CyclicalQuery(0.01, 3, 0.95)
        self.queries[5] = GroupQuery (0.5, self.queries[3], 0.1)
        self.queries[6] = CyclicalQuery (0.01, 4, 0.95)
        #for i in range (len (self.queries)):
        
        self.predictor = QLearn()
    

    def test_predictor (self):
        input = []
        output = []
        
        data = []
        for days in range (0, 7):
            today = []
            for i in range (len (self.queries)):
                today.append (self.queries[i].make_query(days))
            data = data+ today
        
        for days in range (7, 365):
            today = []
            for i in range (len (self.queries)):
                today.append (self.queries[i].make_query(days))
            #print data
            #print today
            #prediction = self.predictor.predict(data)
            input.append (data)
            output.append (today)
            
            #print prediction
            #print today
            data = data+ today
            for i in range (len (self.queries)):
                data.pop (0)
        self.predictor.test_model (input, output)

    def train_data (self):
        input = []
        output = []
        
        data = []
        for days in range (0, 7):
            today = []
            for i in range (len (self.queries)):
                today.append (self.queries[i].make_query(days))
            data = data+ today
        
        
        for days in range (7, 365):
            today = []
            for i in range (len (self.queries)):
                today.append (self.queries[i].make_query(days))
            #print data
            #print today
            input.append (data)
            output.append (today)
            data = data+ today
            for i in range (len (self.queries)):
                data.pop (0)
        self.predictor.set_training_data (input, output)
        self.predictor.train()





if __name__== "__main__":
    test = QTester ()
    test.train_data()
    test.test_predictor()
