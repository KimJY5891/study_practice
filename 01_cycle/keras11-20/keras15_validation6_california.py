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
print(f'x : {x.shape}') # (20640, 8)
print(f'y : {y.shape}') # (20640,)

"""
[실습]
1.train_size 0.7
2. r2 0.55 ~ 6이상
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7, random_state=8715
)
# 2. 모델 구성
model = Sequential()
model.add(Dense(50,input_dim=8))
model.add(Dense(100,activation='relu'))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=740,batch_size=120)

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(f'r2 스코아 : {r2}')

'''
loss : 0.49996742606163025
r2 스코아 : 0.6220199633886054
'''
