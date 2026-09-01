import numpy as np

class GD:
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,W,dW,key):
        return W-self.lr*dW

class Momentum:
    def __init__(self,lr=0.01,beta=0.9):
        self.lr=lr
        self.beta=beta
        self.v={}
    def update(self,W,dW,key):
        if key not in self.v:self.v[key]=np.zeros_like(W)

        self.v[key]=self.v[key]+self.lr*dW
        W=W-self.lr*self.v[key]
        return W

class Adam:
    def __init__(self,lr=0.01,beta1=0.9,beta2=0.99):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.v={}
        self.m={}
        self.t={}
    def update(self,W,dW,key):
        if key not in self.v:self.v[key]=np.zeros_like(W)
        if key not in self.m:self.m[key]=np.zeros_like(W)
        if key not in self.t:self.t[key]=0

        self.t[key]+=1

        self.v[key]=self.beta2*self.v[key]+(1-self.beta2)*(dW**2)
        self.m[key]=self.beta1*self.m[key]+(1-self.beta1)*dW

        m_modif=self.m[key]/(1-np.power(self.beta1,self.t[key]))
        v_modif=self.v[key]/(1-np.power(self.beta2,self.t[key]))
        W=W-((self.lr/np.sqrt(v_modif+1e-10))*m_modif)
        return W