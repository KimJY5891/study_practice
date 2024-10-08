import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])
# 데이터가 많아서 한꺼번에 할 수 없을 수도 있음
# 그래서 쪼개서 훈려해야할 때도 있다. 
# 배치 단위 2라면 데이터 2개씩하고 나머지는 남는대로 훈련
# 단점 : 엄청 느려짐
# 배치가 작게 자르면 성능이 올라감
# 배치의 갯수가 적으면 느려짐 

# 2. 모델 
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size = 1)
# 디폴트 값 = 32
# 배치 값도 결과값에 따라서 다르다.
# 디폴드 값 = 무난한 값 

# 4. 평가 예측 
loss = model.evaluate(x,y)
# x와 y의 값으로 생성된 가중치 w를 넣어서 다시 판단하는것
print("loss : ",loss)
result = model.predict([4])
print("[6]의 예측값 : ",result)
'''
loss :  0.38875025510787964
[6]의 예측값 :  [[4.014045]]
'''
