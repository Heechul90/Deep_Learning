# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옴
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

# 필요한 라이브러리를 불러옴
import pandas as pd
import numpy as np
import tensorflow as tf



#######################################################################
# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 불러오기
raw_data = pd.read_csv('dataset1/wine.csv', header = None)
data = raw_data.copy()
data = data.sample(frac = 1)

# 데이터 내용 확인하기
data.head()
data.info()
data.describe()


# 데이터 나누기
dataset = data.values
X = dataset[:,0:12]
Y = dataset[:,12]

# # 학습셋과 테스트셋을 나눔
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed)


# 모델 설계하기
model = Sequential()
model.add(Dense(30, input_dim = 12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# 모델 컴파일하기
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


# 모델 업데이트하기
import os

MODEL_DIR = 'Study/model1/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "Study/model1/{epoch:02d}-{val_loss:.4f}.hdf5"  # 에폭횟수-오차.hdf5


# 모델 업데이트 및 저장
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',     # 테스트셋의 오차
                               verbose = 1,              # 해당함수의 진행 사항 출력
                               save_best_only = True)    # 오차가 줄어들었을때만 저장


# 학습 자동 중단 설정
from keras.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 100)


# 모델 실행
history = model.fit(X, Y,
                    validation_split = 0.2,
                    epochs = 3500,
                    batch_size = 500,
                    verbose = 0,
                    callbacks = [early_stopping_callback, checkpointer])

# # 테스트를 위해 메모리 내의 모델을 삭제
# del model
#
# # 모델을 새로 불러옴
# model = load_model('Study/model1/1015-0.0393.hdf5')

# 결과 출력
# print('\ Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))


# y_vloss에 테스트셋으로 실험 결과의 오차 값을 저장
y_vloss = history.history['val_loss']


# y_acc에 학습셋으로 측정한 정확도의 값을 저장
y_acc = history.history['accuracy']


# x 값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
import matplotlib.pyplot as plt

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, 'o', c = 'red', markersize = 3)
plt.plot(x_len, y_acc, 'o', c = 'blue', markersize = 3)




