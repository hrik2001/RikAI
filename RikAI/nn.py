import numpy as np

class layer:
        def __init__(self, input_size ,  output_size, f):
                self.f = f.activation
                self.df = f.d
                self.isize = input_size
                self.osize = output_size
                self.param = {}
                #Reason for not defining layer.w and layer.b is so that I can modify w and b later on as passing by reference
                self.param["w"] = np.random.randn(self.isize , self.osize)
                self.param["b"] = np.random.randn(self.osize)
                self.d = {}
        def forward(self, input_matrix):
                self.input_matrix = input_matrix
                return self.f((self.input_matrix @ self.param["w"] + self.param["b"]))
        def backwards(self , dC_dz):
                self.dC_dz=dC_dz
#                print(self.input_matrix.T.size ,(dC_dz * self.df(self.input_matrix @ self.param["w"] + self.param["b"])).size )
                self.d["w"] = self.input_matrix[np.newaxis].T @ (dC_dz * self.df(self.input_matrix @ self.param["w"] + self.param["b"])) 
                #JOEL#self.d["b"] = np.sum((dC_dz * df(self.input @ self.param["w"] + self.param["b"])) , axis=0) # according to me it should be only (dC_dz * df(self.input @ self.param["w"] + self.param["b"]))
                self.d["b"] = dC_dz * self.df(self.input_matrix @ self.param["w"] + self.param["b"])
                return (dC_dz * self.df(self.input_matrix @ self.param["w"] + self.param["b"])) @ self.param["w"].T #returns dC/dz(n-1)

class nn:
        def __init__(self , layers):
                self.layers = layers
        def forward(self,input_matrix):
                for layer in self.layers:
                        input_matrix = layer.forward(input_matrix)
                return input_matrix
        def backwards(self,dC_dz):
                for layer in reversed(self.layers):
                        dC_dz = layer.backwards(dC_dz)
        
