import numpy as np

ACTIVATION_FN={"linear":lambda Z:Z, 
               "sigmoid":lambda Z:1/(1+np.exp(-Z)), 
               "relu":lambda Z:np.maximum(Z,0), 
               "tanh": lambda Z:(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))}
DERIVATIVE={"linear":lambda Z:np.ones_like(Z), 
            "sigmoid":lambda Z:np.exp(-Z)/(1+np.exp(-Z))**2, 
            "relu":lambda Z:(Z>0).astype(float),
            "tanh": lambda Z:1-np.tanh(Z)**2}

def sigmoid(Z):
    exp_Z=np.exp(Z-np.max(Z,axis=1,keepdims=True))
    return exp_Z/np.sum(exp_Z,axis=1,keepdims=True)

def sigmoid_derivative(Z):
    s=sigmoid(Z)
    return s*(1-s)