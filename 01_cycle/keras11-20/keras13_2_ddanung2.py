# 데이콘 따릉이 문제 풀이
import numpy as np
from tensorflow.keras.models import Sequential # Sequential 모델
from tensorflow.keras.layers import Dense # Dense
from sklearn.model_selection import train_test_split
from datetime import datetime
# 대회에선 rmse로 사용
# 우리는 rmse 모르니까 유사지표 사용
from sklearn.metrics import r2_score, mean_squared_error # mse에서 루트 씌우면 rmse로 할수 있을 지도
import pandas as pd
# 우리가 사용할 수 있도록 바꿔야함

# 1. 데이터
# 데이터 불러오기 -> 데이터 분석 -> 결측치 처리 -> dataset 분리
# 경로
path = '/content/drive/MyDrive/_연습/_data/ddarung/'
path_save = '/content/drive/MyDrive/_연습/_save/ddarung/'

# csv 불러오기
train_csv = pd.read_csv(path + 'train.csv',index_col=0)
# 원래라면 아래처럼 하는것
# train_csv = pd.read_csv('/content/drive/MyDrive/_연습/_data/ddarung/train.csv',index_col=0)
# 하지만 변수 path로 인해서 괜찮음
# 불러올 때, read_csv()
# index_col는 인덱스 컬럼이 뭐냐?
# 인덱스는 데이터가 아니니까 매야함
# 컬럼 이름 (header)도 데이터는 아님
# index, header는 연산하지 안는다.
# 인덱스 = 아이디, 색인은 데이터가 아니고 번호 매긴것
print(train_csv)
print(train_csv.shape) # (1459, 10)

test_csv = pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv)
print(test_csv.shape)  # (715, 9)
# count 값이 없다.
#===================================================================================================================
print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(f'train_csv.info : {train_csv.info()}')
# 빨간 점 직전까지 실행
# 결측치 처리
print(f'train_csv.describe : {train_csv.describe()}')
# min : 최소값, max : 최대값, 50% : 중위값
print(f'train_csv type : {type(train_csv)}')
# <class 'pandas.core.frame.DataFrame'>
############################ 결측치 처리 ############################
# 결측치 처리 1. 제거
print(f'train_csv.isnull().sum() : {train_csv.isnull().sum()}')
'''
 hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
'''
train_csv = train_csv.dropna()
# dropna 결측치를 삭제하겠다.
# 변수로 저장하기
print(f'train_csv.isnull().sum() : {train_csv.isnull().sum()}')
'''
train_csv.isnull().sum() :
hour                      0
hour_bef_temperature      0
hour_bef_precipitation    0
hour_bef_windspeed        0
hour_bef_humidity         0
hour_bef_visibility       0
hour_bef_ozone            0
hour_bef_pm10             0
hour_bef_pm2.5            0
count                     0
dtype: int64
'''
print(train_csv.info())
print(f'train_csv.shape : {train_csv.shape}') # (1328, 10)
#####################train_csv 데이터에서 x와 y를 분리#######################

x = train_csv.drop(['count'],axis=1)
# drop : 빼버리겠다. 엑시스 열
# 두 개 이상 리스트
print(f'x.shape : {x.shape}')

y = train_csv['count']
print(f'y.shape : {y.shape}')

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size= 0.7,
    random_state=8715
)
print("x_train.shape : ",x_train.shape) #(929, 9)
print("y_train.shape : ",y_train.shape) #(929, )

# 2. 모델 구성
model = Sequential()
model.add(Dense(15,input_dim=9))
model.add(Dense(30))
model.add(Dense(45))
model.add(Dense(100))
model.add(Dense(56))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=32)

# 4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
# nan = 데이터가 없다.
# 원본 데이터 값이 없어서 nan이 나온다.
# 0은 데이터가 있는 것
# 결측치 값 처리 첫 번째 걍 0으로 처리

y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print(f'r2 스코어 : {r2}')
# rsme 만들기
def rmse(y_test,y_pred) :
  return np.sqrt(mean_squared_error(y_test,y_pred))
# def : 함수의 약자, 함수를 새롭게 정의만 할 때 사용
# 함수란 재사용할 때 사용하는 것
# np.sqrt : 루트

# submission.csv 만들기
print(test_csv.isnull().sum())
''' test_csv에도 결측치 존재
hour                       0
hour_bef_temperature       1
hour_bef_precipitation     1
hour_bef_windspeed         1
hour_bef_humidity          1
hour_bef_visibility        1
hour_bef_ozone            35
hour_bef_pm10             37
hour_bef_pm2.5            36
dtype: int64
'''
y_submit = model.predict(test_csv)
# print(y_submit) # 추측 nan은 결측치 부분인듯
submission = pd.read_csv(path + 'submission.csv',index_col = 0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submit' + datetime.today().strftime("%Y%m%d%H%M%S")+'csv')
'''
loss : 3587.261962890625
r2 스코어 : 0.4970974226663465
'''
