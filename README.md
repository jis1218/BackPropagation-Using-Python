
# Python을 이용한 BackPropagation (특정 값을 구하는 문제)
#### 입력값이 [3, 2, 1], [1, 3, 2], [3, 6, 1], [6, 2, 8]로 주어지고 target 값이 [0.3, 0.5, 1, 0.4]로 주어졌을 때 weight를 갱신하기

#### 신경망의 구조는 다음과 같다.


#### 각 layer는 다음과 같이 함수로 정의하였다.
```python
class Affine(object):

    def __init__(self, W1, W2, b1, b2):
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
        
        
        #transDelta1 = delta1.reshape(2,4)
        self.dW1 = np.dot(self.x1.T, delta1)
        #print(self.dW1)
        self.db1 = np.sum(delta1, axis=0)

    
    def secondForward(self, y1):
        self.y1 =y1
        v2 = np.dot(y1, self.W2) #+ self.b2
    
        return v2
    
    def secondBackward(self, delta2):
        TransitedW2 = self.W2.reshape(1,4)
        Transiteddelta = delta2.reshape(delta2.shape[0],1)
        #dy1 = np.dot(TransitedW2, Transiteddelta)
        #dy1 = np.dot(self.W2, delta2)
        error1 = np.dot(Transiteddelta, TransitedW2)

        self.dW2 = np.dot(delta2, self.y1)
        #print('y1.T', self.y1.T)
        #print('dW2', self.dW2)
        self.db2 = np.sum(delta2, axis=0)
        
        return error1
```
```python
class SigmoidLayer(object):
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
        delta2 = (self.t-self.y2)*(1.0 - self.y2)*self.y2/batch_size      
        
        return delta2
```


##### dW 값을 더하는것인지 빼는 것인지 확인이 필요하다. 어떤 책에는 더하는 것으로 나오고 또 어떤 책은 빼는 것으로 나오는데 차이를 알아야 한다.
```python
twoLayerNet.w1 += learning_rate*dW1 #+를 할지 -를 할지 잘 정해야 한다. 값이 완전히 달라질 수 있다.
```

##### 큰 실수 하나 - batch_mask를 통해 임의의 배치를 epoch 할 때마다 선정해줘야 하는데 나는 돌리기 전에 선정을 해줘서 결국 100개의 배치 내에서만 W값을 구하게 되는 꼴이 된다. 그러기 때문에 test 데이터를 돌리면 낮은 값이 나올 수 밖에 없다.
```python
if __name__ == '__main__':
    
    train_size = x_train.shape[0]
    batch_size = 100
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]

    t_batch = t_train[batch_mask]
    
    W = 0.01*np.random.randn(np.shape(x_batch)[1], np.shape(t_batch)[1])

    
    for i in range(10000):
        
        W = W - learning_rate*logits
    
    pass
```
```python
if __name__ == '__main__':
    
    learning_rate = 0.1
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    train_size = x_train.shape[0]
    batch_size = 100
   
    
    W = 0.01*np.random.randn(x_train.shape[1], np.shape(t_train)[1])
    
    for i in range(10000):
        batch_mask = np.random.choice(train_size, batch_size)

        x_batch = x_train[batch_mask]

        t_batch = t_train[batch_mask]

        W = W - learning_rate*logits
    
    pass
```
##### 모멘텀을 이용했더니 학습률이 현저하게 좋아졌다. 모멘텀을 적용하지 않았을 경우 특정 accuracy에 도달하기까지 대략 500번 걸리던 것이 모멘텀을 사용하니 100번 정도 걸렸다.
```python
    momentum_rate = 0.9
    momentum = 0
    momentum = momentum*momentum_rate - learning_rate*np.dot(x_batch.T, dx) 
    W = W + momentum
    #W = W - learning_rate*np.dot(x_batch.T, dx) 모멘텀을 사용하지 않았을 경우의 코드
```