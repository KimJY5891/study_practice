# x의 전체 값을 잘라서 train과 test값을로 만들 수 있다. 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,])
y=np.array([10,9,8,7,6,5,4,3,2,1,])

#[실습] numpy 리스트의 슬라이싱!! 7:3으로 잘라라
x_train = x[0:7]
y_train = y[0:7]
x_test = x[7:10]
y_test = y[7:10]

# 2. 모델 구성 
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(12))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=10)

# 4. 평가 예측 
loss = model.evaluate(x_test,y_test) # 트레인 값음 없음. 당분간 테스트 값은 훈련에 사용하지 않는다.
print('loss : ',loss)

result = model.predict([[11]])
print('[11]의 예측값은',result)
'''
loss :  46.0118408203125
[11]의 예측값은 [[9.92539]]
'''
