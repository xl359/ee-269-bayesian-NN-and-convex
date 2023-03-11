#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:59:19 2023

@author: LiuXinwei
"""
# https://towardsdatascience.com/from-theory-to-practice-with-bayesian-neural-network-using-python-9262b611b825
import torch
from torch import nn
import torchbnn as bnn
import numpy as np
import torch.utils as utils
import torch.optim as optim
import matplotlib.pyplot as plt



# calculate x, y, and the dimension 

def data_generation(n_radii,n_angles):
    
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]
    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())

    x_plot = x
    y_plot = y

    n = len(x) #number of sample
    x = np.reshape(x,(n,1))
    y = np.reshape(y,(n,1))

    x = np.hstack((x,y))
    y = np.sin(-x[:,0]*x[:,1]) + 0.05*np.random.rand(n)
    z_plot = y

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot_trisurf(x_plot, y_plot, z_plot, linewidth=0.2, antialiased=True)

    plt.show()
    return n,x,y, radii,angles,z_plot
    
 # --------------------------specify common parameters ------------------------------
epochs = 2000
m = 1000 # dimension of D and hidden nodes
lr=0.01
beta = 1e-100
sigma = 1e-3
eta = 1e-3
beta = 1e-10
np.random.seed(42)
d = 2
percent_train = 0.7

# ---------------------------------original data this set prediction is good-------------------------------------- 
np.random.seed(10)
n=50
d=3
x=np.random.randn(n,d-1)
x=np.append(x,np.ones((n,1)),axis=1)

y=((np.linalg.norm(x[:,0:d-1],axis=1)>1)-0.5)*2


# ---------------------------------complex data set-------------------------------------- 
# d = 2
# np.random.seed(42)
# n_radii = 8
# n_angles = 36
# n,x,y, radii,angles,z_plot = data_generation(n_radii,n_angles)


# ---------------------------------data split-------------------------------------- 

x_train = x[:int(n*percent_train),:]
y_train = y[:int(n*percent_train)]
x_test = x[int(n*percent_train):,:]
y_test = y[int(n*percent_train):]
# ---------------------------------Training-------------------------------------- 


train_x = torch.Tensor(x_train)
train_y = torch.Tensor(y_train)
train_y = torch.unsqueeze(train_y, dim=1)
dataset = utils.data.TensorDataset(train_x, train_y)
dataloader = utils.data.DataLoader(dataset)


n_input = d
n_hidden = m
n_output = 1


model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=sigma, in_features=n_input, out_features=n_hidden),
                      nn.ReLU(),
                      bnn.BayesLinear(prior_mu=0, prior_sigma=sigma, in_features=n_hidden, out_features=n_output))


optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
loss_plot = []
for step in range(epochs):
    train_x = train_x.view(train_x.shape[0], -1)
    output = model(train_x)
    # https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param)
        
    loss = loss_fn(output, train_y) + beta*l2_reg
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_plot.append(loss.item())
    print(step)

models_result = np.array([model(train_x).data.numpy() for k in range(n)])
models_result = models_result[:,:,0]    
models_result = models_result.T
mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
std_values = np.array([models_result[i].std() for i in range(len(models_result))])

plt.figure(figsize=(10,8))
plt.plot(loss_plot)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(mean_values,color='navy',lw=3,label='Predicted Mean Model')
plt.fill_between(np.linspace(0, len(mean_values)-1,len(mean_values)),mean_values-3.0*std_values,mean_values+3.0*std_values,alpha=0.2,color='navy',label='99.7% confidence interval')
plt.plot(y,'.',color='darkorange',markersize=4,label='Test set')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
    


# ---------------------------------Testing-------------------------------------- 


test_x = torch.Tensor(x_test) 
test_y = torch.Tensor(y_test)
test_y = torch.unsqueeze(test_y, dim=1)

models_result = np.array([model(test_x).data.numpy() for k in range(n)])
models_result = models_result[:,:,0]    
models_result = models_result.T
mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
std_values = np.array([models_result[i].std() for i in range(len(models_result))])

plt.figure(figsize=(10,8))
plt.plot(loss_plot)
plt.show()



plt.figure(figsize=(10,8))
plt.plot(np.linspace(int(n*percent_train),n-1,n - int(n*percent_train)),mean_values,color='navy',lw=3,label='Predicted Mean Model')
plt.fill_between(np.linspace(int(n*percent_train),n-1,n - int(n*percent_train)),mean_values-3.0*std_values,mean_values+3.0*std_values,alpha=0.2,color='navy',label='99.7% confidence interval')
plt.plot(y,'.',color='darkorange',markersize=4,label='Test set')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
    


