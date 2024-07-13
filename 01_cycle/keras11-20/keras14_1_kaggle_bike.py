from sklearn.metrics import r2_score, mean_squared_log_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
# 여러 데이터를 읽기 쉽다.

# 1. 데이터 
# 경로
path = '/content/drive/MyDrive/_연습/_data/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv',index_col=0)
print(f'train_csv : {train_csv}')
print(f'train_csv.shape : {train_csv.shape}') # (10886, 11)

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(f'test_csv : {test_csv}')
print(f'test_csv.shape : {test_csv.shape}') # (6493, 8)
print(f'train_csv : {train_csv.columns}')
print(f'test_csv : {test_csv.columns}')
'''
train_csv : Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
test_csv : Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
      dtype='object')
test.csv에는 'casual', 'registered', 'count'가 없는 상태임
'''

# 결측치 처리
print(f'train isnull sum : {train_csv.isnull().sum()}') # 0개
print(f'test isnull sum : {test_csv.isnull().sum()}') # 0개
print('train_csv.info() : ',train_csv.info())
train_csv = train_csv.dropna()

# x와 y를 분리
x = train_csv.drop(['count'],axis=1)
print(f'x.shape : {x.shape}') # (10886, 10)
y = train_csv['count']
print(f'y.shape : {y.shape}') # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state = 8715
)
print(f'x_train.shape : {x_train.shape}, x_test.shape : {x_test.shape}')
print(f'y_train.shape : {y_train.shape}, y_test.shape : {y_test.shape}')

# 2. 모델 구성
model = Sequential()
model.add(Dense(15,input_dim=10))
model.add(Dense(30,activation='relu'))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련  
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=900,batch_size=100,verbose=1)

# 4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
'''
마이너스가 나오는 이유
처음에 랜덤하게 선을 긋고 시작해서 
활성화 함수 
한정화 함수
나중에 다룸 
relu는 0이상의 값은 양수로 유지 0 이하의 값은 0이 되는 함수 
output dim 히든레이어에 넣는다. 
activation= 'relu'
'''
loss : 0.0007426271331496537
'''

'''
