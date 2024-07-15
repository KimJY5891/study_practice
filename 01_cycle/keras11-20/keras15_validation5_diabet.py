#diabet:당뇨병
#어느 데이터셋인지도 알아내야함

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

'''
[실습]
1. train_size : 0.7 ~ 0.9
2. r2 0.62 이상
2번 교육용 데이터 인데도 그만큼 잘 안맞기도한다.
실무용 데이터는 더 안맞을 수도 있다.
잘 맞게하려면 데이터 정제를 잘해야한다.
'''

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print("x:",x.shape) # (442, 10)
print("y:",y.shape) # (442,)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9, random_state=555
)

# 2. 모델 구성 
model = Sequential()
model.add(Dense(20,input_dim=10))
model.add(Dense(12))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# 노드나 레이어가 늘어날 수록 훈련 값을 더 높여야 잘 나오는 것으로 추측된다. 

# 3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1220,batch_size=72)

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ',loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print('r2스코어 : ',r2)
# 실습데이터라 데이터 전처리 되어있음, 나중에는 우리가 데이터 전처리 해야줘야함
'''
loss:  2620.4208984375
r2스코어 :  0.6178644246974505
'''





