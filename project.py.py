import numpy as np
import pandas as pd
import copy,math
import matplotlib.pyplot as plt

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0.0
    for i in range (m):
        f_wb=np.dot(x[i],w)+b
        cost=cost+(f_wb-y[i])**2
    cost=cost/(2*m)
    return cost


def compute_gradient(x,y,w,b):
    m,n=x.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        f_wb=np.dot(x[i],w)+b
        err=f_wb-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+(err*x[i,j])
        dj_db=dj_db+err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db



def gradient_decent(x,y,w_in,b_in,compute_cost,compute_gradient,alpha,num_iters):
    j_history=[]
    w=copy.deepcopy(w_in)
    b=b_in
    for i in range(num_iters):
        dj_dw,dj_db=compute_gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        if (i<100000):
            j_history.append(compute_cost(x,y,w,b))
        if i%math.ceil(num_iters/10)==0:
            print("Iteration ",i," cost ",j_history[-1])
    return w,b,j_history
    
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b_init = 785.1811367994083
initial_w=np.zeros_like(w_init)
initial_b=0
iteration=1000
alpha=5.0e-7

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])



cost=compute_cost(x_train,y_train,w_init,b_init)
print("The cost is ",cost)


tmp_dj_dw,tmp_dj_db=compute_gradient(x_train,y_train,w_init,b_init)
print("The value of dj_dw and dj_db are ",tmp_dj_dw, " ",tmp_dj_db)


w_final,b_final,j_hist=gradient_decent(x_train,y_train,initial_w,initial_b,compute_cost,compute_gradient,alpha,iteration)

print("b and w found by gradient decent are ",b_final," ", w_final)