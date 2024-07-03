# 훈련 데이터로 평가하지 않는다.
# 객관적인 평가를 하기 위해서는 훈련에 사용한 데이터는 사용하면 안된다.
# 우리가 수집한 데이터를 나눈다. 훈련에 사용한 데이터 / 평가용 데이터로 나눈다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,])
y=np.array([10,9,8,7,6,5,4,3,2,1,]) #,옆에 아무것도 적지 않아도 에러가 나지 않는다.
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# 2. 모델 구성 
model = Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=10)

# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

result = model.predict([[11]])
print('[11]의 예측값은',result)
'''
loss :  0.0006656329496763647
[11]의 예측값은 [[10.962197]]
'''
