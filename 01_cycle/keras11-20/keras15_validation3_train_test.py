import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(
    x_train,y_train,
    train_size = 0.7, random_state=8715,shuffle = False
)
x_test, x_val, y_test, y_val = train_test_split(
    x_test,y_test,
    train_size=0.5,random_state=8715,shuffle=False
)
print(x_test) 
print(x_val)
print(y_test)
print(y_val)

# 2. 모델 구성 
model = Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=100,
          validation_split=0.2
          )

# 4. 평가 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

y_pred = model.predict([17])
print('17의 예측값 : ',y_pred)
'''
loss :  4.547473508864641e-12
17의 예측값 :  [[17.]]
'''
