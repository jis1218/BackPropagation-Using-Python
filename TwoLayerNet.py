'''
Created on 2018. 3. 20.

@author: Insup Jung
'''
import numpy as np
from BackPropagation.AllClass import *

class TwoLayerNet(object):
    

    def __init__(self):
        self.w1 = np.array([[0.4, 0.6, 0.2, 0.8], [0.6, 0.2, 0.7, 0.3], [0.8, 0.1, 0.4, 0.5]])
        self.w2 = np.array([0.2, 0.5, 0.1, 0.6])
        self.b1 = np.array([0.1, 0.2, 0.5, 0.2])
        self.b2 = np.array([0.1, 0.2, 0.5, 0.2])
        
        self.dW1 = None
        self.dW2 = None
        self.db1 = None
        self.db2 = None
        #self.soft = SoftmaxWithLoss()
        self.affine1 = Affine(self.w1, self.b1)
        self.affine2 = Affine(self.w2, self.b2)
        self.relu = ReLULayer()
        self.sigmoid = SigmoidLayer()
    def predict(self, x):
        
        y = self.affine1.forward(x)        
        y = self.sigmoid.forward(y)
        y = self.affine2.forward(y)
        
        return y
    
    def getResult(self):
        return self.sigmoid.result()
        
    def loss(self, x, t):
        y = self.predict(x)
        
        y = self.sigmoid.secondforward(y, t)
        return y
    
    def gradient(self, x, t):
        
        self.loss(x, t)
        
        dout = 1
        dout = self.sigmoid.lastBackward()
        dout = self.affine2.backward(dout)
        dout = self.sigmoid.backward(dout)
        dout = self.affine1.backward(dout)
        
        return self.affine1.dW, self.affine2.dW, self.affine1.db, self.affine2.db
            
    
    def firstLayer(self, x):
        output = np.dot(x, self.w1)
        relu = ReLULayer()
        output = relu.forward(output)
        
        return output
    
    def secondLayer(self, x):
        output = np.dot(x, self.w2)        
        return output
    
    #
        
        
        
    

        