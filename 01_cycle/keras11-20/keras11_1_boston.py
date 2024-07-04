from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
# 교육용 데이터셋
from sklearn.datasets import load_boston

# 1. 데이터

datasets = load_boston()
# 정규화 : 예를 들어서 1에서 1조 데이터 일 때, 1조x1조 일 경우, 오버 쿨럭이 걸릴수도 있음
# 그래서 0부터 1사이로 압축 => 최대치를 낮춘다.

x = datasets.data
y = datasets.targe
print('feature_names')
print('x : ', x)
print('y : ', y)
# 워닝 = 경고는 하지만 실행은 된다.
# 에러 = 실행 안됨
#'feature_names': array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
#       'TAX', 'PTRATIO', 'B', 'LSTAT']  'B' - 그 당시는 문제 없었지만 지금은 인종차별적인 문제가 있다. 
print(datasets.DESCR)
#INSTANCE(506) -예시라는 말은 데이터라는 것도 의미할수도 있다. 
#ATTRIBUTE(13) - 속성 - 특성, 열
#MEDV = 결과값 =Y 단위 천달라
print("x:",x.shape)#(506,13)
print("y:",y.shape)#(506,)(벡터는 1개)
x_train, x_test,y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=1234
)
# 2 모델 구성
model = Sequential()
model.add(Dense(26,input_dim=13))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(26))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))
# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=32)
# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss: ',loss)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print('r2스코어 : ',r2)

