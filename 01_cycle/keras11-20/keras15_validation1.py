import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x_train =np.array(range(1,11))
y_train =np.array(range(1,11))
print(x_train)
print(y_train)
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
print(x_val)
print(y_val)
# 훈련하고 검증
# 훈련하고 검증
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])

# 2. 모델 구성 
model=Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(100,))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=100,
          validation_data=[x_val,y_val]
          )

# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')

y_pred = model.predict([17])
print(f'17의 예측값은 {y_pred}')
'''
loss : 0.001026501995511353
17의 예측값은 [[16.934233]]
'''
