
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
# 교육용 데이터셋
from sklearn.datasets import fetch_california_housing

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(f'x : {x.shape}') # x : (20640, 8)
print(f'y : {y.shape}') # y : (20640,)
"""
[실습]
1.trainsize 0.7
2. r2 0.55 ~ 6이상
"""

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.7,
    random_state=1234
)
# 2. 모델 구성 
model = Sequential()
model.add(Dense(4,input_dim=8))
model.add(Dense(2))
model.add(Dense(8))
model.add(Dense(80))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1200,batch_size=1000)
# 4. 평가 예측 
loss = model.evaluate(x_test,y_test)
print('loss: ',loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print('r2스코어 : ',r2)
'''
loss:  0.6458858251571655
194/194 [==============================] - 1s 2ms/step
r2스코어 :  0.5153553365604124
'''
