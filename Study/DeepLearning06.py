# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옴
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

# 필요한 라이브러리를 불러옴
import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf


# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train.shape

#
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# 결과를 0과 1로 바꿔주는 원핫인코딩
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)



# 컨볼루션 신경망 설정
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))


# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# 모델 최적화 설정
MODEL_DIR = 'Study/model1/'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "Study/model1/{epoch:02d}-{val_loss:.4f}.hdf5"

# 모델 업데이터
checkpointer = ModelCheckpoint(filepath = modelpath,     # 위에서 지정한 경로
                               monitor = 'val_loss',     # 테스트셋의 오차
                               verbose = 1,              # 학습과정 보여주기
                               save_best_only = True)    # 오차가 적어졌을때만 기록


# 학습 자동 중단
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 10)


# 모델의 실행
history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    epochs = 30,
                    batch_size = 200,
                    verbose = 0,
                    callbacks = [early_stopping_callback, checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


# 테스트셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')


# 그래프에 그리드를 주고 레이블을 표시

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()