# 캐글 
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

# 여러 데이터를 읽기 쉽다. 

# 1. 데이터 
# 경로
path = '/content/drive/MyDrive/_연습/_data/kaggle_bike/'
path_save = '/content/drive/MyDrive/_연습/_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(f'train_csv.shape : {train_csv.shape}') # (10886, 11)
print(f'test_csv.shape : {test_csv.shape}') #  (6493, 8)
# 결측치 처리
print(f'train.isnull().sum() : { train_csv.isnull().sum()}')
print(f'test.isnull().sum() : {test_csv.isnull().sum()}')


# x와 y로 분리
x= train_csv.drop(['casual','registered','count'],axis=1) 
y = train_csv['count']

# train,test 셋 분리
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715
)


# 2. 모델 구성 
model = Sequential()
model.add(Dense(8,input_dim=8))
model.add(Dense(16))
model.add(Dense(48))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=100)

# 4. 평가 예측 
# 평가 
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
# 예측
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print(f' r2 : {r2}')

def rmse(y_test,y_predict) :  #함수 정의만 한 것 
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse=rmse(y_test,y_pred)
print("rmse : ",rmse)

# submission.csv 만들기
y_submit = model.predict(test_csv)
# count에 넣기 
submission = pd.read_csv(path + 'sampleSubmission.csv',index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submit' + datetime.today().strftime("%Y%m%d%H%M%S") +'.csv')
'''
loss : 25436.88671875
r2 : 0.2642263416649875
rmse :  159.48944202650748
'''
