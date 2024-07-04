from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
# 교육용 데이터셋 
from sklearn.datasets import load_boston

# 1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets.target
print('feature_names')
print("x:",x)
print("y:",y)
print(datasets.DESCR)
print("x:",x.shape)#(506,13)
print("y:",y.shape)#(506,)(벡터는 1개)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size = 0.7,
    random_state= 1234
)
# 2. 모델 구성 
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
model.fit(x_train,y_train,epochs=1000,batch_size=100,verbose=1)
# 506이 행이고, 배치 사이즈 1일 때 한 에포당 345로 나옴
# 이유는 트레인 사이즈가 0.7이기 때문이다.
# 사람한테 상태를 보여주기 위해서 딜레이가 생김 ( 사람 눈이 빠르게 인식을 못해서)
# 훈련 시키고 보여주고 하는 것을 보고 싶지 않으면 
# 0 : 부정 아무것도 안나옴 
# 1와 auto : 긍정 다 보여줌 
# 2 : 진행바만 안나옴
# verbose의 디폴드 값 : 1

# 4. 평가 예측 
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score

r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
