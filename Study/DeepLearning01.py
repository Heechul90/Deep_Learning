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

# 환자의 기록과 수술 결과를 X와 Y로 구분
X = data_set[:, 0:17]
Y = data_set[:, 17]

# 딥러닝 구조를 결정(모델을 설정하고 실행하는 부분)
model = Sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 딥러닝을 실행
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X, Y, epochs = 30, batch_size = 10)
