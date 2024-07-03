#x의 전체 값을 잘라서 트레인과 테스트 값으로 만들 수 있다.
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,])
y=np.array([10,9,8,7,6,5,4,3,2,1,])
# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 
# 힌트 사이킷런 - 트레인 테스트 split
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    # train_size와 test_size 합쳐서 100프로를 넘으면 안된다.
    train_size = 0.7,
    random_state = 42, # 랜덤값에 대한 고정
    # random_state 랜덤하게 값을 뺄 때, 완전 랜덤하지 않다.
    # 똑같은 값이여야 모델 비교가 되는데 완전 랜덤이면 어렵다.
    shuffle=True
    #  디폴트 True
)
print(f'x_train : {x_train},x_test : {x_test}') 
print(f'y_train : {y_train},y_test : {y_test}') 
# x_train : [ 1  8  3 10  5  4  7],x_test : [9 2 6]
# y_train : [10  3  8  1  6  7  4],y_test : [2 9 5]

# 2. 모델 구성 
model=Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100)

# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

result = model.predict([[11]])
print('result : ',result)
'''
loss :  0.3648759424686432
result :  [[0.8536201]]
'''
