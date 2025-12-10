from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def normalize(data):
    m = data.shape[0]
    n = data.shape[1]
    
    new_data = np.zeros((m,n))
    
    for i in range(n):
        temp = data[:,i]
        new_data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])
    
    return new_data

def ptp(data):
    m = data.shape[0]
    n = data.shape[1]
    
    res = np.zeros(n)
    
    for i in range(n):
        min_val = data[0,i]
        max_val = data[0,i]
        for j in range(m):
            min_val = min(data[j,i],min_val)
            max_val = max(data[j,i],max_val)
        res[i] = max_val-min_val
    
    return res
  
def cost(X,y,w,b):
    m = X.shape[0]
    total_cost = 0
    
    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        cost = (f_wb - y[i])**2
        total_cost += cost
    
    total_cost /= (2*m)
    return total_cost

def gradient(X,y,w,b):
    m = X.shape[0]
    n = X.shape[1]
    
    dj_dw = np.zeros(n)
    dj_db = 0 
    
    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += (err*X[i,j]/m)
        dj_db += err
    dj_db /= m 
    return dj_dw,dj_db

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,iter_num):
    w = w_in
    b = b_in
    
    J_hist = []
    J_hist.append(cost(X,y,w,b))
    
    for i in range(iter_num):
        dj_dw , dj_db = gradient_function(X,y,w,b)
        
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
        J_hist.append(cost_function(X,y,w,b))
    
    return w,b,J_hist

def predict(x,w,b):
    return np.dot(x,w)+b

data = pd.read_csv("data.csv",delimiter=",",dtype=float).to_numpy()

X_train,y_train = data[:,0:-1],data[:,-1]
X_train = normalize(X_train)

initial_w = np.zeros(X_train.shape[1])
initial_b = 0

new_parameters = gradient_descent(X_train,y_train,initial_w,initial_b,cost,gradient,0.001,10000)
w = new_parameters[0]
b = new_parameters[1]

print(cost(X_train,y_train,w,b))
print(w,b)

#fig, axes = plt.subplots(1, 2)

plt.plot(new_parameters[2])
# axes[0].set_xscale("log")
# axes[0].set_xscale("log")
plt.xlabel("No. of iterations")
plt.ylabel("Cost")

# axes[1].scatter(X_train.flatten(),y_train,s=2)
# x  = np.linspace(-2,3,2)
# y  = w[0]*x + b
# axes[1].plot(x,y,color="red")

plt.show()