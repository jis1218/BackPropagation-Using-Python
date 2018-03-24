# coding: utf-8
'''
Created on 2018. 3. 20.

@author: Insup Jung
'''
import numpy as np
from BackPropagation.AllClass import *

class TwoLayerNet(object):
    

    def __init__(self):
        self.w1 = np.array([[0.3, 0.5, 0.8, 0.7], [0.2, 0.2, 0.2, 0.4], [0.1, 0.6, 0.4, 0.2]])
        self.w2 = np.array([0.5, 0.5, 0.5, 0.5])
        self.b1 = np.array([0.5, 0.3, 0.2, 0.6])
        self.b2 = np.array([0.2, 0.5, 0.1, 0.2])
        
        self.dW1 = None
        self.dW2 = None
        self.db1 = None
        self.db2 = None
        #self.soft = SoftmaxWithLoss()
        self.affine = Affine(self.w1, self.w2, self.b1, self.b2)

        #self.relu = ReLULayer()
        self.sigmoid = SigmoidLayer()
    def predict(self, x):
        #print('x value', x)
        v1 = self.affine1.forward(x)
        #print('conv value', v1)        
        y1 = self.sigmoid.forward(v1)
        #print('first sigmoid', y1)  
        v2 = self.affine2.forward(y1)
        #print('second conv', v2)  
        
        return v2
    
    def getResult(self):
        return self.sigmoid.result()
        
    def loss(self, x, t):
        v2 = self.predict(x)
        
        y2 = self.sigmoid.secondforward(v2, t)
        #print('second sigmoid', y2)
        return y2
    
    def gradient(self, x, t):
        
        #print('x value', x)
        v1 = self.affine.firstForward(x)
        #print('first conv v1', v1)        
        y1 = self.sigmoid.forward(v1)
        #print('first sigmoid y1', y1)  
        v2 = self.affine.secondForward(y1)
        #print('second conv v2', v2)  
        y2 = self.sigmoid.secondForward(v2, t)
        #print('second sigmoid y2', y2)
        dout = 1
        delta2 = self.sigmoid.secondBackward()
        #print('second sigmid backward delta2', delta2)
        dout = self.affine.secondBackward(delta2)
        #print('second conv backward error1', dout)
        delta1 = self.sigmoid.backward(dout, y1)
        #print('first sigmoid backward delta1', delta1)
        dout = self.affine.firstBackward(delta1)
        #print('first conv backward', dout)
        
        return self.affine.dW1, self.affine.dW2, self.affine.db1, self.affine.db2
            
    
    def firstLayer(self, x):
        output = np.dot(x, self.w1)
        relu = ReLULayer()
        output = relu.forward(output)
        
        return output
    
    def secondLayer(self, x):
        output = np.dot(x, self.w2)        
        return output
    
    #
        
        
        
    

        