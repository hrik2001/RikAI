import numpy as np

class SE:
    def __init__(self , correct , predicted):
        self.correct = correct
        self.predicted = predicted
    def loss(self):            
        return ((self.correct - self.predicted)**2)
    def d(self):
        return 2*(self.correct - self.predicted)
