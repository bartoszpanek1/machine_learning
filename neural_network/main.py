from neural_network.n_net import NeuralNetwork
import numpy as np
from scipy.io import loadmat

a=loadmat('ex3data1.mat',squeeze_me=False)


X,y=a["X"], a["y"]
data=np.asarray(X,dtype=np.float32)
labels=np.asarray(y,dtype=np.int32)

b=loadmat('ex3weights.mat')
Theta1,Theta2=b["Theta1"], b["Theta2"]

print(data.shape)
print(labels.shape)
print(Theta1.shape)
print(Theta2.shape)
print('-----------------------')
nn=NeuralNetwork(3,[400,25,10],0.1)
nn.matrices=[Theta1,Theta2]
p=nn.predict(data)

correct_answ=0
m=data.shape[0]

for i in range(0,m):
    if (p[i]+1)==labels[i][0]:
        correct_answ=correct_answ+1


print(correct_answ/m)
