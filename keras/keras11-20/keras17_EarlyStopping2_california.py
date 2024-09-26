# 빨리(Early) 끊는다.(Stop)
# 로스가 최소 지점이 최적의 웨이트
# history 사용하여 컷 시키기
# 소문자는 함수, 대문자는 클래스
# 네이밍 룰 c언어, python(c언어로 만들었음), 처음이랑 띄어쓰기 대문자 사용(C언어 계열), _를 사용하는 경우(java계열)
# 카멜케이스 룰
# 강제적인 것은 아님
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
datasets= fetch_california_housing()
x=datasets.data
y=datasets['target']
print(f'x.shape : {x.shape}. y.shape : {y.shape}')  #(506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.7, random_state = 8715
)

# 2. 모델구성
model = Sequential()
model.add(Dense(4,input_dim=8))
model.add(Dense(2))
model.add(Dense(8))
model.add(Dense(80))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
                   verbose=1,restore_best_weights=True
)# 브레이크 잡은 시점에서 가중치를 사용해서 예측을 실행함 ( 복원한다고 표현)
# verbose = 끊는 지점을 볼 수 있음
# val_loss가 나음
# patience = 몇 번까지 참을지
# mode = min - 최소 값을 찾아라
hist = model.fit(x_train,y_train,
          epochs=1200,batch_size=120,
          validation_split=0.2,verbose=1,
          callbacks=[es])
print("---------------------------------------------")
print("hist:",hist)
print("---------------------------------------------")
print(hist.history)
print("---------------------------------------------")
print(hist.history['loss'])
print("---------------------------------------------")
print(hist.history['val_loss'])

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['val_loss'],marker='.',c='red',label='val_loss')
plt.plot(hist.history['loss'],marker='.',c='blue',label='loss')
plt.title('캘리포니아') #이름 지어주기,한글로 쓰면 깨진다. #한글로 제대로 나오게 하는 방법이 있다.
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend() #
plt.grid() #
plt.show()


# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print(f'r2 스코어 : {r2_score}')

