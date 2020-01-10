# 딥러닝을 구동하는 데 필요한 케라스 함수를 불러옴
from keras.models import Sequential
from keras.layers import Dense

# 필요한 라이브러리를 불러옴
import numpy as np
import tensorflow as tf
import pandas as pd

# 데이터 불러오기
raw_data = pd.read_csv('dataset1/pima-indians-diabetes.csv',
                       names = ['pregnant', 'plasma', 'pressure', 'thickness',
                                'insulin', 'BMI', 'pedigree', 'age', 'class'])
data = raw_data.copy()


# 데이터 내용 확인하기
data.head()
data.info()
data.describe()
data[['pregnant', 'class']]
data[['plasma', 'class']]


# 데이터 가공하기
data[['pregnant', 'class']].groupby(['pregnant'], as_index = False).mean().sort_values(by = 'pregnant', ascending = True)



# 데이터 시각화하기
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (12, 12))
sns.heatmap(data.corr(),
            linewidths = 0.1,
            vmax = 0.5,                 # 색상의 밝기를 조절하는 인자
            cmap = plt.cm.gist_heat,    # 미리 정해진 matplotlib 색상의 설정값
            linecolor = 'white',
            annot = True)
plt.show()


# plasma(포도당 부하 검사 2시간 후 공복 혈당 농도(mm Hg))와 class와 관계
grid = sns.FacetGrid(data, col = 'class')
grid.map(plt.hist, 'plasma', bins = 10)
plt.show()



#######################################################################
# seed 값 생성
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기
dataset = np.loadtxt('dataset1/LottoMachineLearningDataSet1.csv',
                     delimiter = ',')

dataset.shape
X = dataset[:, 0:7]
Y = dataset[:, 7]


# 모델 설정
# 입력층 8
# 은닉층 노드 12개, 활성화 함수: 렐루
# 은닉층 노드 8개, 활성화 함수: 렐루
# 출력층 노드 1개, 활성화 함수: 시그모이드
model = Sequential()
model.add(Dense(30, input_dim = 7, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y, epochs = 200, batch_size = 10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))



