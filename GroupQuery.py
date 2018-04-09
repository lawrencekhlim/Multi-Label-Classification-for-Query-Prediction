from Query import Query
import random
class GroupQuery (Query):
    def __init__ (self, chance, query, group_chance):
        Query.__init__ (self, chance)
        self.paired_query = query
        self.group_chance = group_chance
    
    '''
    def __init__ (self, chance, group_chance):
        Query.__init__ (self, chance)
        self.period = period
        self.group_chance = group_chance
    '''
    def pair_query (self, query):
        self.paired_query = query
    
    def make_query (self, day):
        if day in self.dict:
            return self.dict [day]
        roll = random.random()
        if (self.paired_query.make_query(day) == 1 and roll < self.group_chance):
            self.dict [day] = 1
            return 1
        elif (roll < self.probability):
            self.dict [day] = 1
            return 1
        else:
            self.dict [day] = 0
            return 0
