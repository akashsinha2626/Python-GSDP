import pandas as pd
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt
x=pd.read_csv("gsdp.csv")
data=np.array(x)
data=np.delete(data,0,1)
data=data/data.max(axis=0)
t = np.linspace(0,1,10)
# Model
input = data[:,1:7]
output = data[:,0:1]
# plot
plt.figure()
plt.scatter(input[:,0], output)
plt.scatter(input[:,1], output)
plt.scatter(input[:,2], output)
plt.scatter(input[:,3], output)
plt.scatter(input[:,4], output)
plt.scatter(input[:,5], output)
plt.xlabel('GSDP')
plt.ylabel('AG,AAG,ID,MANF,CON,SV')
plt.title('Data')
plt.show()
# Neural Network
nn = nl.net.newff([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],[6,5,1])
# Traning
nn.trainf = nl.train.train_gd
# Train the Neural network
error = nn.train(input, output, epochs=1000,show= 500, goal = 0.01)
print("Minimum Values Of Error")
print(min(error))
#plot
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
# run the Neural network
opt = nn.sim(input)
ypd = output.reshape(10)
plt.figure()
plt.plot(t,output,'-',t,output,'.')
plt.show()
# Predict
a=0.9
b=0.8
c=0.8
d=0.9
g=0.9
f=0.8
prd=np.array([a,b,c,d,g,f])
prd=prd.reshape(1,6)
p=nn.sim(prd)
print(p)


