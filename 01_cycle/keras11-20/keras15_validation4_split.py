import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터 
x =np.array(range(1,17))
y =np.array(range(1,17))

x=np.array(range(1,17)) #(10,)
y=np.array(range(1,17)) 

x_train=np.array([14,15,16])
y_train=np.array([14,15,16])

x_test=np.array([11,12,13])
y_test=np.array([11,12,13])

# 2. 모델 구성 
model = Sequential()
model.add(Dense(100,input_dim=1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=100,
          validation_split=0.2)

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')

result = model.predict([17])
print('17의 예측값 : ',result)
'''
loss : 0.00022853408881928772
17의 예측값 :  [[16.940596]]
'''
