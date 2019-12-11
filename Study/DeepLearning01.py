# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옴
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옴
import numpy as np
import tensorflow as tf

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 준비된 수술 환자 데이터를 불러옴
data_set = np.loadtxt('dataset1/ThoraricSurgery.csv',
                      delimiter = ',')
data_set
data_set.shape


# 환자의 기록과 수술 결과를 X와 Y로 구분
X = data_set[:, 0:17]
Y = data_set[:, 17]


# 딥러닝 구조를 결정(모델을 설정하고 실행하는 부분)
# 딥러닝은 퍼셉트론 위에 숨겨진 퍼셉트론 층을 차곡차곡 추가하는 형태
# 층들을 케라스에서 Sequential() 함수를 통해서 구현
# model.add를 통해서 라인(층)을 추가
# Dence 함수를 통해서 구체적으로 구조를 정함

model = Sequential()                                       # Sequential() 함수를 model로 선언
model.add(Dense(30, input_dim = 17, activation = 'relu'))  # model.add로 층을 추가, Dense 함수로 30개의 노드생성
model.add(Dense(1, activation = 'sigmoid'))                #


# 딥러닝 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
# epochs: 학습 프로세스가 모든 샘플에 대해 한 번 실행되는 것을 1epoch
# batch_size: 샘플을 한번에 몇 개씩 처리할지를 정하는 부분
model.fit(X, Y, epochs = 30, batch_size = 10)
