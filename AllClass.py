# coding: utf-8
'''
Created on 2018. 3. 22.

@author: Insup Jung
'''

import numpy as np
from common.functions import * 

class ReLULayer(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.mask = None
        
    
    def forward(self, x):
        self.mask = (x <=0) #mask에는 x의 형상대로 true 또는 false가 들어간다.
        out = x.copy()
        out[self.mask]=0 #out중 true인 것에만 0을 넣어준다. numpy array의 특징
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Affine(object):
    '''
    classdocs
    '''


    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x =x
        out = np.dot(x, self.W) + self.b
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx
    
class SigmoidLayer(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.out = None
        self.t = None
    
    def forward(self, x):
        out = 1 / (1+np.exp(-x))
                
        return out
    
    def secondforward(self, x, t):
        out = 1 / (1+np.exp(-x))
        self.out = out
        self.t = t
        
        return out
    
    def result(self):
        return self.out
    
    def backward(self, dout):
        dx = dout*(1.0 - self.out)*self.out
        
        return dx
        
    def lastBackward(self):
        dx = (self.t-self.out)*(1.0 - self.out)*self.out        
        
        return dx
    
class SoftmaxWithLoss(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def result(self):
        return self.y
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)
        
        return dx
        