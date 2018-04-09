from Query import Query
import random
class ChainQuery (Query):
    def __init__ (self, chance, chained_chance):
        Query.__init__ (self, chance)
        self.chained_chance = chained_chance
        self.previous = False
    def make_query (self, day):
        if day in self.dict:
            return self.dict [day]
        roll = random.random()
        if (self.previous and roll < self.chained_chance):
            self.dict [day] = 1
            return 1
        elif (roll < self.probability):
            self.dict [day] = 1
            self.previous = True
            return 1
        else:
            self.dict [day] = 0
            self.previous = False
            return 0
