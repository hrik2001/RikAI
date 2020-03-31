import numpy as np

class SGD:
        def __init__(self, nn , lr=0.01):
                self.nn = nn
                self.lr=lr
        def optimize(self):
                for layer in self.nn.layers:
                        #print(layer.param , layer.d)
                        #print(layer.param["w"].shape , layer.d["w"].shape)
                        #print(layer.param["b"].shape , layer.d["b"].shape)
                        layer.param["w"] += self.lr * layer.d["w"].reshape(layer.param["w"].shape)
                        layer.param["b"] += self.lr * layer.d["b"].reshape(layer.param["b"].shape)
