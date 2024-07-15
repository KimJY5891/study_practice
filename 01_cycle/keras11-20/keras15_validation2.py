from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터 
x_train =np.array(range(1,17)) 
y_train =np.array(range(1,17))

x_val = x_train[13:]
y_val = y_train[13:]
x_test = x_train[10:13]
y_test = x_train[10:13]

# 2. 모델구성
model = Sequential()
model.add(Dense(15, input_dim=1))
model.add(Dense(30))
model.add(Dense(45))
model.add(Dense(75))
model.add(Dense(45))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=64,
          validation_data=[x_val,y_val]
          )

# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')

y_pred = model.predict([17])
print('17의 예측값 : ',y_pred)
'''
loss : 0.0007310239598155022
17의 예측값 :  [[17.010624]]
'''
