import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터 
datasets= load_breast_cancer()
print(datasets)

'''
사전에서 무언가를 찾는다.
딕셔너리 = 사전
키밸류 형태로 저장한 사전 : 딕셔너리
중괄호 형태로 되어잇다. 
'''
'''
print(datasets.DESCR) - 판다스 : .describe() - 찾아보기
열 이름 보기 - 판다스 : .columns
판다스는 많이 사용하기 때문에 이해하기
'''
x=datasets['data']
y=datasets.target
# 위에 둘 다 딕셔너리의 키 
print(x.shape,y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size = 0.8, random_state=8715,
    shuffle=True
)
# 2. 모델 구성
model=Sequential()
model.add(Dense(100,input_dim=30))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1,activation='sigmoid'))
# 마지막에 sigmoid 넣는다. 
# 최종값을 0과 1로 사이로 한정시키다.
# activateion = 활성화 함수 

# 3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse']) # mean+squard_error, acc  값 확인용 훈련에 영향을 미치지 않는다. 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
                   verbose=1,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=100,batch_size=132,validation_split=0.2,
                 callbacks=[es])
# 회귀에서는 매트릭에서 입력해서 볼 수 있음 

# 4. 평가, 예측
result =model.evaluate(x_test,y_test)
# loss와 메트릭스에 집어넣은 값이 들어간다.
print('result : ',result )

y_pred = model.predict(x_test)

# accuracy_socre : 정확도 지표
# 0이냐 1이냐
# 서로 같냐를 따지는 지표 
print('==================================================')
print(f' y_test : {y_test[:5]}') 
print(f'y_pred[:5] :{y_pred[:5]}')
print(f'np.round(y_pred[:5]) : {np.round(y_pred[:5])}')
'''
 y_test : [1 1 1 0 1] # 1101
y_pred[:5] ::[[1.       ][0.9999999][1.       ][0.       ][1.       ]] # 실수형태 
np.round(y_pred[:5]) : [[1.][1.][1.][0.][1.]] # 반올림 
이진 분류와 연속된 숫자가 독같이 처리할 수 없어서 반올림해도됨
'''
acc = accuracy_score(y_test,np.around(y_pred))
print('acc : ',acc)
'''
acc = accuracy_score(y_test,y_pred)
ValueError: Classification metrics can't handle a mix of binary and continuous targets
-> 분류 메트릭은 이진숫자와 실수와 섞일 수 없다는 의미 
acc = accuracy_score(y_test,np.around(y_pred))
로 변경해주면 된다. 더이상 실수가 아니기 때문임 
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='blue',label='loss') 
plt.title('breast_cancer') #이름 
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()
