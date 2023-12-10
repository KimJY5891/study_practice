
# 1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100)

# 4. 평가, 예측
loss = model.evaluate(x,y)
# x와 y의 값으로 생성된 가중치(w)를 넣어서 다시 판단하는 것 
print('loss :',loss)
result = model.predict([4])
print('[4]의 예측값 : ',result)
# loss : 0.00972045212984085
# [4]의 예측값 :  [[3.8124228]]
