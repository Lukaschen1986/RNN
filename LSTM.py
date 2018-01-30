# -*- coding: utf-8 -*-
# https://yq.aliyun.com/articles/174270
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import keras
from keras.layers.convolutional import Conv1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.advanced_activations import PReLU
from keras import backend as K
K.image_data_format()
K.set_image_data_format('channels_first')
from keras import initializers
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model, model_from_json, model_from_yaml
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
#import spacy
import matplotlib.pyplot as plt

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

reframed = pd.DataFrame(scaled)
reframed["new"] = reframed.iloc[:,0].shift(-1)
reframed.columns = ["var1(t-1)","var2(t-1)","var3(t-1)","var4(t-1)","var5(t-1)","var6(t-1)","var7(t-1)","var8(t-1)","var1(t)"]

# split into train and test sets
values = reframed.values
values = values[0:-1,:]
n_train_hours = 365 * 24
train = values[0:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, 0:-1], train[:, -1]
test_X, test_y = test[:, 0:-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
inpt = Input(shape=(train_X.shape[1], train_X.shape[2]))
x = LSTM(units=50)(inpt)
x = Dense(1)(x)

model = Model(inputs=inpt, outputs=x)
bs = 64; epc = 50; lr = 0.1; dcy = 0.04
#lr*(1-dcy)**np.arange(epc)
#model.compile(loss="mean_squared_error", optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=10**-8), metrics=["mae"]) # decay=dcy
model.compile(loss="mean_squared_error", optimizer=SGD(lr=0.1, momentum=0.9, decay=0.1), metrics=["mae"])
#early_stopping = EarlyStopping(monitor="loss", patience=2, mode="auto", verbose=1)
model_fit = model.fit(train_X, train_y, batch_size=bs, epochs=epc, verbose=2, shuffle=False, validation_data=(test_X, test_y)) # , callbacks=[early_stopping]

plt.plot(model_fit.history['loss'], label='train')
plt.plot(model_fit.history['val_loss'], label='test')


LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)

GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
