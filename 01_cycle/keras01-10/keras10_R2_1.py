import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 넘파이 사용하기 전에만 임포트하면 됨
# 굳이 위에 작성할 필요는 없다. 
import matplotlib.pyplot as plt

# 1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,6,17,23,21,20])
x_train,x_test,y_train,y_test = train_test_split(
    x, # x_train과 x_test로 분리
    y, # y_train과 y_test로 분리
    train_size = 0.7,
    shuffle = True,
    random_state = 1234
) # train_ size는 0.6 ~ 0.9 권장

# 2. 모델 구성 
model = Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=100)
# 4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
y_pred = model.predict(x_test)
# 훈련 데이터 말고 테스트 용으로 평가 해야한다.

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print('r2 스코어 : ', r2)

'''
loss :  0.04356187954545021
r2 스코어 :  0.9986303688308475
'''
