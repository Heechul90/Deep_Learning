import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 데이터 불러오기
raw_data = pd.read_csv('data/temperature.csv')
raw_data

data = raw_data.copy()
data.columns
data.head()

data = data.set_index('time')
data.head()


