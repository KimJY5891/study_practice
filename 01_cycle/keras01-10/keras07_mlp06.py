#x는 3개, y는 3개
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10),range(21,31),range(201,211)])
print(x.shape)
x = x.transpose()
print(x.shape)

y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
            [9,8,7,6,5,4,3,2,1,0]
            ]) # (3,10)
print(y.shape)
y = np.transpose(y)
print(y.shape)

#실습 예측 : [9,30,210]-[10,1.9,0]
# 2. 모델 구성 
model = Sequential()
model.add(Dense(7,input_dim=3)) # x의 열의 갯수 
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(3))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

# 4. 평가 예측
loss = model.evaluate(x,y)
print('loss : ',loss)

result = model.predict([[9,30,210]])
print('[9,30,210]의 예측값은 ',result)
'''
loss :  11.871042251586914
[9,30,210]의 예측값은  [[5.071506  0.8684078 8.079336 ]]
'''
