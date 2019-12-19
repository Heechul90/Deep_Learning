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
raw_data = pd.read_csv('dataset1/sonar.csv', header = None)
data = raw_data.copy()


# 데이터 내용 확인하기
data.head()
data.info()
data.describe()

dataset = data.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]
X.shape


# 원-핫 인코딩
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 학습셋과 테스트셋을 나눔
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed)

X_train.shape
X_test.shape

# 모델 설계
# 입력층: 노드 4개
# 은닉층1: 노드 16개, 렐루
# 은닉층2: 노드 8개, 렐루
# 출력층: 노드3개, 소프트맥스
model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


# 모델 실행
model.fit(X_train, Y_train,
          epochs = 130,
          batch_size = 5)

# 모델을 컴퓨터에 저장
model.save('Study/my_model.h5')


# 테스트를 위해 메모리 내의 모델을 삭제
del model

# 모델을 새로 불러옴
model = load_model('Study/my_model.h5')


# 결과 출력
print('\ Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))
