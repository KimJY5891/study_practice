# 데이콘 따릉이 문제 풀이 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential # model
from tensorflow.keras.layers import Dense
# 우리는 rmse 모르니까 유사지표 사용 
from sklearn.metrics import r2_score, mean_squared_error 
# 대회에서 rmse로 사용 
# mse에서 루트 씌우면 rmse로 할 수 있을 지도
# 우리가 사용할 수 있도록 바꿔야함 

# 1. 데이터 
path = '/content/drive/MyDrive/_연습/_data/ddarung/'
# ./  현재 폴더, 상대경로 

train_csv = pd.read_csv(path+'train.csv',index_col=0)
# index_col은 인덱스 컬럼이 무엇인지 
# 인덱스는 데이터가 아니니까 빼야함 
# 인덱스 = 아이디.
# 색인(index)은 데이터가 아니라 번호 매긴것 
#print(train_csv)
print(train_csv.shape) # (1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)
print(test_csv.shape) # (715, 9) # count 값이 없다. 
#===================================================================================================================
print(train_csv.columns)  
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info())
# <class 'pandas.core.frame.DataFrame'>
# Index: 1459 entries, 3 to 2179
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    1459 non-null   int64  
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
# memory usage: 125.4 KB
# None

# 결측치 처리해야한다.
# 빨간 점 직전까지 실행 
print(train_csv.describe())
# min : 최소값, max : 최대값, 50% : 중위값
print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>
####################################### 결측치 처리 #######################################                                                                                   #
# 결측치 처리 1. 제거 
print(train_csv.isnull())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# dropna : 결측치를 삭제하겠다. 
# 변수로 저장하기 
print(f'train_csv.isnull().sum() : {train_csv.isnull().sum()}')
print(f' train_csv.info() : {train_csv.info()}')
print(f' train_csv.shape : {train_csv.shape}')
'''
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   hour                    1328 non-null   int64  
 1   hour_bef_temperature    1328 non-null   float64
 2   hour_bef_precipitation  1328 non-null   float64
 3   hour_bef_windspeed      1328 non-null   float64
 4   hour_bef_humidity       1328 non-null   float64
 5   hour_bef_visibility     1328 non-null   float64
 6   hour_bef_ozone          1328 non-null   float64
 7   hour_bef_pm10           1328 non-null   float64
 8   hour_bef_pm2.5          1328 non-null   float64
 9   count                   1328 non-null   float64
dtypes: float64(9), int64(1)
memory usage: 114.1 KB
 train_csv.info() : None
 train_csv.shape : (1328, 10)
'''


####################################### 결측치 처리 #######################################                                                                                   #

#####################train_csv 데이터에서 x와 y를 분리#######################
x = train_csv.drop(['count'],axis=1)
print(f'x.shape : {x.shape}') # x.shape : (1328,9)
# drop : 빼버리겠다. 엑시스 열 
# 두 개 이상 리스트
y = train_csv['count']
print(f'y.shape : {y.shape}') # y.shape : (1328,)
#####################train_csv 데이터에서 x와 y를 분리#######################

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size = 0.7, random_state = 517
)
print(f'x_train.shape : {x_train.shape}') # y.shape : (1328,)
print(f'x_test.shape : {x_test.shape}') # y.shape : (1328,)
print(f'y_train.shape : {y_train.shape}') # y.shape : (1328,)
print(f'y_test.shape : {y_test.shape}') # y.shape : (1328,)
'''
x_train.shape : (929, 9)
x_test.shape : (399, 9)
y_train.shape : (929,)
y_test.shape : (399,)
'''

# 2. 모델 구성 
model = Sequential()
model.add(Dense(55, input_dim=9))
model.add(Dense(55))
model.add(Dense(55))
model.add(Dense(55))
model.add(Dense(55))
model.add(Dense(55))
model.add(Dense(55))
model.add(Dense(1)) # y 값은 한개

# 3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1500,batch_size=100)
# 4. 예측 평가 
loss = model.evaluate(x_test,y_test)
print('loss: ',loss)
# nan = 데이터가 없다. 원본 데이터 값이 없어서 nan이 나온다. 
# 0은 데이터가 있는것 
# 결측치 값 처리 첫번째 걍 0으로 처리
y_pred = model.predict(x_test)
print(f'r2 : {r2_score(y_test,y_pred)}')
'''
loss:  2760.9052734375
13/13 [==============================] - 0s 2ms/step
r2 : 0.542444293653913
'''
