import numpy as np

class tanh:
	def activation(self, x):
		return np.tanh(x)
	def d(self,x):
		y = np.tanh(x)
		return 1 - y ** 2
