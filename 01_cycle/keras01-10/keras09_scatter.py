
import numpy as np
import matplotlib.pyplot as plt # matplotlib : 그래프를 그릴 수 있는 라이브러리
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size = 0.3,
    shuffle = True,
    random_state =1234,
) 
# x =  x_train, x_test
# y = y_train, y _test로 분류 된다.
# x,y 위치가 바뀌어도 괜찮음 

# 2. 모델 구성 
model = Sequential()
model.add(Dense(6, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=10)

# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)
y_pred = model.predict(x)
print("y_pred : ", y_pred)
print(y.shape)
