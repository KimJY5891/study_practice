import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([
    range(10),
    range(21,31),
    range(201,211)
]) # (10,3)
print(x)
'''
[[  0   1   2   3   4   5   6   7   8   9] 
 [ 21  22  23  24  25  26  27  28  29  30] 
 [201 202 203 204 205 206 207 208 209 210]]
'''
print(x.shape) # (10,3)
x = x.transpose()
print(x.shape) # 

y=np.array([[1,2,3,4,5,6,7,8,9,10]]) #(1,10)
y=y.T
print(y) 


# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=3)) # x의 열의 갯수
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1)) # y의 열의 갯수 만큼 아웃풋 됨 

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=60,batch_size=10)

# 4. 평가 예측
loss = model.evaluate(x,y)
print("loss :", loss)
result = model.predict([[9, 30, 210]])
print("[[9, 30, 210]]의 예측값 :", result)
'''
loss : 0.8619085550308228
[[9, 30, 210]]의 예측값 : [[9.733274]]
'''
