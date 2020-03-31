from RikAI.nn import layer , nn
from RikAI.optimizer import SGD
from RikAI.loss import SE
from RikAI.activation import tanh

#import pickle

import numpy as np

inputs =[
np.array([0,0]),
np.array([1,0]),
np.array([0,1]),
np.array([1,1])
]

outputs =[
np.array([[1]]),
np.array([[0]]),
np.array([[0]]),
np.array([[1]])
]


input_output = zip(inputs , outputs)
a = []
for i in input_output:
    a.append((i[0], i[1]))
input_output = a


net = nn([layer(2 , 20, tanh()),
layer(20 ,10  , tanh()),
layer(10 , 1 , tanh())
])

epochs = 5000
for i in range(epochs):
    print("EPOCH ",i+1)
    loss_net=0
    for j in input_output:
        answer = net.forward(j[0])
        loss = SE(j[1] , answer)
        loss_net += (np.sum(loss.loss()))
        net.backwards(loss.d())
        opm = SGD(net)
        opm.optimize()
    print("LOSS::" , loss_net)   
    print("##############################################")

#a = open("checkpoint" , "wb")
#pickle.dump(net , a)
#a.close()

print("According to the AI")
print("0 XOR 1 is ", net.forward(np.array([0,1])))
print("0 XOR 0 is ", net.forward(np.array([0,0])))
print("1 XOR 1 is ", net.forward(np.array([1,1])))
print("1 XOR 0 is ", net.forward(np.array([1,0])))
