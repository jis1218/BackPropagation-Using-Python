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

    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.x1 = None
        self.dW1 = None
        self.db1 = None
        self.W2 = W2
        self.b2 = b2
        self.y1 = None
        self.dW2 = None
        self.db2 = None
        
    def firstForward(self, x1):
        self.x1 =x1
        v1 = np.dot(x1, self.W1) + self.b1
        
        return v1
    
    def firstBackward(self, delta1):
        #dx1 = np.dot(dout, self.W1.T)
        transX = self.x1.reshape(3, 1)
        transDelta1 = delta1.reshape(1,4)
        self.dW1 = np.dot(transX, transDelta1)
        print(self.dW1)
        self.db1 = np.sum(delta1, axis=0)

    
    def secondForward(self, y1):
        self.y1 =y1
        v2 = np.dot(y1, self.W2) + self.b2
    
        return v2
    
    def secondBackward(self, delta2):
#         TransitedW2 = self.W2.reshape(4,1)
#         Transiteddelta = delta2.reshape(1,4)
#         dy1 = np.dot(TransitedW2, Transiteddelta)
        #dy1 = np.dot(self.W2, delta2)
        error1 = self.W2*delta2

        self.dW2 = self.y1*delta2
        #print('y1.T', self.y1.T)
        print('dW2', self.dW2)
        self.db2 = np.sum(delta2, axis=0)
        
        return error1
    
class SigmoidLayer(object):
    '''
    classdocs
    '''


    def __init__(self):
        self.y2 = None
        self.t = None
    
    def forward(self, v1):
        y1 = 1 / (1+np.exp(-v1))
                
        return y1
    
    def secondForward(self, v2, t):
        y2 = 1 / (1+np.exp(-v2))
        self.y2 = y2
        self.t = t
        
        return y2
    
    def result(self):
        return self.y2
    
    def backward(self, error1, y1):
        delta1 = error1*(1.0 - y1)*y1
        
        return delta1
        
    def secondBackward(self):
        batch_size = self.t.shape[0]
        
        delta2 = (self.t-self.y2)*(1.0 - self.y2)*self.y2      
        
        return delta2
    
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
        