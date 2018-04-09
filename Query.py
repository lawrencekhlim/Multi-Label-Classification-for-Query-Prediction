import random
class Query:

    def __init__(self, chance):
        self.probability = chance
        self.dict = {}
    def make_query (self, day):
        if day in self.dict:
            return self.dict [day]
        if (random.random() < self.probability):
            self.dict [day] = 1
            return 1
        else:
            self.dict [day] = 0
            return 0
