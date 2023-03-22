#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 03:34:24 2023

@author: LiuXinwei
"""



import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def drelu(x):
    return x>=0


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



def dimentioanl_mapping(n,m,x):
    D=np.empty((n,0))

    for i in range(int(m)):
        u=np.random.randn(d,1)
        D=np.append(D,drelu(np.dot(x,u)),axis=1)

    return D

def Theta_X_compute(n,d,m,D,x):
    ThetaX_pos = np.empty((x.shape[0],d))
    for i in range(m):
        Di = np. diag(D[:,i])
        Dxi = Di@x
        ThetaX_pos = np.concatenate((ThetaX_pos, Dxi), axis=1)
        
    ThetaX_pos  = ThetaX_pos[:,d:]
    ThetaX  = np.concatenate((ThetaX_pos, -ThetaX_pos), axis=1)
    return ThetaX

def convex_solve(D,d,m,n,x,y):

    v = cp.Variable((d, m)) # d by m
    w = cp.Variable((d, m))
    constraints = []
    obj_a = cp.sum(cp.multiply(D , (x@(v - w))), axis=1)
    cost=cp.sum(cp.pos(1-cp.multiply(y,obj_a)))/n + beta*(cp.mixed_norm(v.T,2,1)+cp.mixed_norm(w.T,2,1))
    constraints += [cp.multiply(2*D - np.ones((n,m)), (x@v)) >= 0]
    constraints += [cp.multiply(2*D - np.ones((n,m)), (x@w)) >= 0]
    obj_val = cp.Minimize(cost)
    prob = cp.Problem(obj_val, constraints)
    prob.solve()
    cvx_opt=prob.value
    print("Convex program objective value (eq (8)): ",cvx_opt)
    return v,w



 # --------------------------specify common parameters ------------------------------
sigma = 1e-3
eta = 1e-3
#eta = eta**2
m = 1000# dimension of D and hidden nodes
lr = 0.01
beta = 1e-10#1
np.random.seed(42)
percent_train = 0.7


# ---------------------------------original data this set prediction is not good but works-------------------------------------- 
# np.random.seed(10)
# n=50
# d=3
# x=np.random.randn(n,d-1)
# x=np.append(x,np.ones((n,1)),axis=1)

# y=((np.linalg.norm(x[:,0:d-1],axis=1)>1)-0.5)*2

# ---------------------------------complex three d data--------------------------------------
 
#access data from data generation   
d = 2
np.random.seed(42)
n_radii = 8
n_angles = 36
n,x,y, radii,angles,z_plot = data_generation(n_radii,n_angles)

# ---------------------------------data split--------------------------------------

D = dimentioanl_mapping(n,m,x)

# train data calculation
x_train = x[:int(n*percent_train),:]
y_train = y[:int(n*percent_train)]
D_train = D[:int(n*percent_train),:]
ThetaX_train = Theta_X_compute(n,d,m,D_train,x_train)

# test data calculation 
x_test = x[int(n*percent_train):,:]
y_test = y[int(n*percent_train):]
D_test = D[int(n*percent_train):,:]
ThetaX_test = Theta_X_compute(n,d,m,D_test,x_test)



# ---------------------------------code for solving the cvx--------------------------------------


v,w = convex_solve(D_train,d,m,int(n*percent_train),x_train,y_train)


# ---------------------------------code for computing the posterior--------------------------------------


v = v.value.flatten()
v = np.reshape(v, (len(v), 1))
w = w.value.flatten()
w = np.reshape(w, (len(w), 1))

W = np.vstack((v,w))

Sigma_inv = sigma**(-2)*ThetaX_train.T@ThetaX_train + 1/eta*np.identity(ThetaX_train.shape[1])


Sigma = np.linalg.inv(Sigma_inv)


mu = sigma**(-2)*Sigma@ThetaX_train.T@y_train


#mean_values = ThetaX@mu


std_values = []
mean_values = []
for i in range(int(n*percent_train)):
    mean_valuesi = mu.T@ThetaX_train[i,:]
    mean_values.append(mean_valuesi)
    std_valuesi = np.sqrt(ThetaX_train[i,:].T@Sigma@ThetaX_train[i,:] + sigma**2)
    std_values.append(std_valuesi)

std_values = np.reshape(std_values,(int(n*percent_train),1))
#mean_values = np.reshape(mean_values,(n,1))
#mean_values = ThetaX@mu

plt.figure(figsize=(10,8))
plt.plot(mean_values,color='navy',lw=3,label='Predicted Mean Model')
plt.fill_between(np.linspace(0, len(mean_values)-1,len(mean_values)),mean_values-3.0*std_values.T[0],mean_values+3.0*std_values.T[0],alpha=0.2,color='navy',label='99.7% confidence interval')
plt.plot(y,'.',color='darkorange',markersize=4,label='Test set')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

    



# ---------------------------------TESTING-----------------------------------------------------------------------------------------



# ---------------------------------get the thetaX and calculate the corresponding posterior--------------------------------------
std_values = []
mean_values = []
for i in range(n - int(n*percent_train)):
    mean_valuesi = mu.T@ThetaX_test[i,:]
    mean_values.append(mean_valuesi)
    std_valuesi = np.sqrt(np.sqrt(ThetaX_test[i,:].T@Sigma@ThetaX_test[i,:] + sigma**2))
    std_values.append(std_valuesi)

std_values = np.reshape(std_values,(n - int(n*percent_train),1))

plt.figure(figsize=(10,8))
plt.plot(np.linspace(int(n*percent_train),n-1,n - int(n*percent_train)),mean_values,color='navy',lw=3,label='Predicted Mean Model')
plt.fill_between(np.linspace(int(n*percent_train),n-1,n - int(n*percent_train)),mean_values-3.0*std_values.T[0],mean_values+3.0*std_values.T[0],alpha=0.2,color='navy',label='99.7% confidence interval')
plt.plot(y,'.',color='darkorange',markersize=4,label='Test set')
plt.legend()
plt.xlabel('data points')
plt.ylabel('y')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(mean_values,color='navy',lw=3,label='Predicted Mean Model')
plt.fill_between(np.linspace(0, len(mean_values)-1,len(mean_values)),mean_values-3.0*std_values.T[0],mean_values+3.0*std_values.T[0],alpha=0.2,color='navy',label='99.7% confidence interval')
plt.plot(y_test,'.',color='darkorange',markersize=4,label='Test set')
plt.legend()
plt.xlabel('data points')
plt.ylabel('y')
plt.show()

