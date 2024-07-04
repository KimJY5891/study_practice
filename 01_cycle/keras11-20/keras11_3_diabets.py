

#diabet:당뇨병
# 어느 데이터셋인지도 알아내야함 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
# 대회 문제는 이정도
"""
[실습]
1.trainsize 0.7~0.9
2.r2 0.62 이상
2번은 그만큼 교육용 데이터인데도 잘 안맞기도한다. 
실무용 데이터는 더 안맞을 수도 있다. 
잘맞게하려면 데이터 정제를 잘해야한다. 
"""
# 1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state = 1234
)
print(f'x : {x.shape}')
print(f'x_train : {x_train.shape}')
'''
x : (442, 10)
x_train : (309, 10)
'''
print(f'y : {y.shape}')
print(f'y_train : {y_train.shape}')
'''
y : (442,)
y_train : (309,)
'''
# 2. 모델 구성 
model = Sequential()
model.add(Dense(10,input_dim=10))
model.add(Dense(30))
model.add(Dense(150))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=1000)

# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(f'r2 : {r2}')
'''
loss : 2863.8916015625
r2 : 0.4863554304435366
'''
