import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 1. 데이터
x = np.array([ 
    [1,1],
      [2,1],
      [3,1],
      [4,1],
      [5,2],
      [6,1.3],
      [7,1.4],
      [8,1.5],
      [9,1.6],
      [10,1.4]]
   
)

# 10행 2열 
# 행 : 데이터셋의 갯수, 열 : 특성 
# 행 무시, 열 우선 
# 모델링을 할 때 열의 갯수를 판단하게 될 것이다.
# 열, feature, 특성, 컬럼

y = np.array([11,12,13,14,15,16,17,18,19,20])

# 2. 모델 구성
model = Sequential()
model.add(Dense(3,input_dim=2))
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(18))
model.add(Dense(22))
model.add(Dense(25))
model.add(Dense(18))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mse',optimizer = 'adam')
model.fit(x,y,epochs=1100,batch_size=5)

# 4. 평가 예측
loss = model.evaluate(x,y)
# x와 y의 값으로 생성된 가중치 w를 넣어서 다시 판단한느것 
print("loss : ",loss)



result= model.predict([[10,1.4]])# 괄호 숫자를 잘 맞춰야함
print("[10,1.4]의 예측값 : ",result)
# loss :  4.179946611060359e-09
# [10,1.4]의 예측값 :  [[20.000072]]
