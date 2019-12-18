# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옴
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옴
import numpy as np
import tensorflow as tf
import pandas as pd

# 데이터 불러오기
raw_data = pd.read_csv('dataset1/iris.csv',
                       names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
data = raw_data.copy()


# 데이터 내용 확인하기
data.head()
data.info()
data.describe()

# 데이터 시각화하기
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data, hue = 'species')
plt.show()


#######################################################################
# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 불러오기
raw_data = pd.read_csv('dataset1/iris.csv',
                       names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
data = raw_data.copy()

dataset = data.values
dataset.shape


X = dataset[:, :4].astype(float)
Y = dataset[:, 4]


# 원-핫 인코딩
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y)
Y = e.transform(Y)

# 0, 1로 변환
from keras.utils import np_utils
Y_encoded = np_utils.to_categorical(Y)


# 모델 설계
# 입력층: 노드 4개
# 은닉층1: 노드 16개, 렐루
# 은닉층2: 노드 8개, 렐루
# 출력층: 노드3개, 소프트맥스
model = Sequential()
model.add(Dense(16, input_dim = 4, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))


# 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


# 모델 실행
model.fit(X, Y_encoded,
          epochs = 50,
          batch_size = 1)


# 결과 출력
print('\ Accuracy: %.4f' % (model.evaluate(X, Y_encoded)[1]))