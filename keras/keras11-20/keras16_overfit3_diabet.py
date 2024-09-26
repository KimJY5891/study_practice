#diabet:당뇨병
#어느 데이터셋인지도 알아내야함

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
#대회문제도 이정도 

# 1. 데이터 
datasets= load_diabetes()
x= datasets.data
y= datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9, random_state = 8715
)
print("x:",x.shape) # (442, 10)
print("y:",y.shape) # (442,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100,input_dim=10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train,y_train,
                 epochs=1200,batch_size=120,
                 validation_split=0.2,verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)




import matplotlib.pyplot as plt

plt.plot(hist.history['val_loss'],marker='.',c='red',label='val_loss') 
plt.plot(hist.history['loss'],marker='.',c='blue',label='loss')
plt.title('diabet') #이름 지어주기
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()

