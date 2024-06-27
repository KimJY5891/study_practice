# 1. 데이터
import numpy as np
x = np.array([1,2,3,]) # 파이썬 리스트에서 1차원 NumPy 배열을 생성
y = np.array([1,2,3,])
# loss는 작을 수록 좋다. 

# 2.모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
# 텐서플로우 안에 카라스 안에 모델스 안에 Sequential를 가지고 와라 
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(1,input_dim=1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=30)
# fit = 훈련시키다 
# epochs = 훈련양
# 훈련양이 너무 많아도 loss값이 이상하게 될 수 있다. 
# loss가 가장 작은 지저에서 끊어야한다. 
# 똑같은 값으로 훈련시켜도 처음에 랜덤하게 시작하기 때문에 다른 결과가 나타날 수 있다. 
#loss: 16.0596
