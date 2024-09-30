import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
path = "/content/drive/MyDrive/_연습/_data/dacon_diabets/"
path_save = "/content/drive/MyDrive/_연습/_save/dacon_diabets/"
train_csv = pd.read_csv(path+'train.csv',index_col=0)
test_csv = pd.read_csv(path+'test.csv',index_col=0)
####################################### 결측치 처리 #######################################                                                                                 #
print(train_csv.isnull().sum())
'''
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
'''
'''
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
'''
print(train_csv.info())
'''
# 함수를 이용하여 데이터 df의 정보를 출력함
Index: 652 entries, TRAIN_000 to TRAIN_651
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               652 non-null    int64
 1   Glucose                   652 non-null    int64
 2   BloodPressure             652 non-null    int64
 3   SkinThickness             652 non-null    int64
 4   Insulin                   652 non-null    int64
 5   BMI                       652 non-null    float64
 6   DiabetesPedigreeFunction  652 non-null    float64
 7   Age                       652 non-null    int64
 8   Outcome                   652 non-null    int64
dtypes: float64(2), int64(7)
memory usage: 50.9+ KB
None
'''
print(train_csv.shape) # (652, 9)
####################################### 결측치 처리 #######################################                                                                                 #
x = train_csv.drop(['Outcome'],axis=1)
y = train_csv['Outcome']
print("x.shape : ",x.shape) # x.shape :  (652, 8)
print("y.shape : ",y.shape) # y.shape :  (652,)

x_train, x_test, y_train, y_test= train_test_split(
    x,y,
    train_size = 0.7, shuffle = True, random_state=8715
)

# 2. 모델 구성
model =Sequential()
model.add(Dense(800,input_dim =8))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(1,activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',
              metrics=['accuracy','mse',])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=12, mode='max',
    verbose=1,restore_best_weights=True
)
model.fit(x_train,y_train,epochs = 128, batch_size=128, verbose=1,
          validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
result =model.evaluate(x_test,y_test)
print('result : ',result )
y_pred = np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_pred)
print(f'acc : {acc}')

y_submit = np.round(model.predict(test_csv))
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['Outcome'] =y_submit
submission.to_csv(path_save+'submit'+time.strftime('%Y%m%d', time.localtime(time.time())))
print('완료')
