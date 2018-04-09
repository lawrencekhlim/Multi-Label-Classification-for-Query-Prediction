from Query import Query
import random
class CyclicalQuery (Query):
    def __init__ (self, chance, period, periodic_chance):
        Query.__init__ (self, chance)
        self.period = period
        self.periodic_chance = periodic_chance
    def make_query (self,day):
        if day in self.dict:
            return self.dict [day]
        roll = random.random()
        if (day%self.period == 0 and roll < self.periodic_chance):
            self.dict [day] = 1
            return 1
        elif (roll < self.probability):
            self.dict [day] = 1
            return 1
        else:
            self.dict [day] = 0
            return 0

