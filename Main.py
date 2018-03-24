# coding: utf-8
'''
Created on 2018. 3. 20.

@author: Insup Jung
'''

import numpy as np
from BackPropagation.TwoLayerNet import *

if __name__ == '__main__':
    
    x = np.array([[3, 2, 1], [1, 3, 2], [3, 6, 1], [6, 2, 8]])
    t = np.array([0.3, 0.5, 1, 0.4])
    twoLayerNet = TwoLayerNet()
    learning_rate = 0.8
    
    for i in range(100000):
        dW1, dW2, db1, db2 =  twoLayerNet.gradient(x, t)       
        twoLayerNet.w1 += learning_rate*dW1 #+를 할지 -를 할지 잘 정해야 한다. 값이 완전히 달라질 수 있다.
        twoLayerNet.w2 += learning_rate*dW2
        twoLayerNet.b1 += learning_rate*db1
        twoLayerNet.b2 += learning_rate*db2        
    print(twoLayerNet.getResult()) #이 줄의 의미는 정답일 확률이 다음과 같다는 것
    
    pass