# -*- coding: utf-8 -*-
# https://yq.aliyun.com/articles/174270
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.convolutional import Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
from matplotlib import pyplot

parser = lambda x: datetime.strptime(x, '%Y %m %d %H')
#dataSet = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv", index_col=0)
dataSet = pd.read_csv("PRSA_data_2010.1.1-2014.12.31.csv", index_col=0, parse_dates=[["year","month","day","hour"]], date_parser=parser)
dataSet = dataSet.drop("No", axis=1)
dataSet.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataSet.index.name = "date"
dataSet["pollution"].fillna(value=0, inplace=True)
dataSet = dataSet[24:]

values = dataSet.values
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataSet.columns[group], y=0.5, loc='right')
	i += 1
#pyplot.show()

encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4]) # 风速特征是标签编码（整数编码）。如果你有兴趣探索，也可以使用热编码。
values = values.astype("float32")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

df = pd.DataFrame(scaled)
df["t+1"] = df.iloc[:,0].shift(-1)
