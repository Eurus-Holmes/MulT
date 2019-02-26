# RNN处理时序数据的二分类


import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from keras import backend as K
import my_callbacks
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
max_lenth = 23
max_features = 12
training_iters = 2000
train_batch_size = 800
test_batch_size = 800
n_hidden_units = 64  
lr = 0.0003
cb = [
    my_callbacks.RocAucMetricCallback(), # include it before EarlyStopping!
    EarlyStopping(monitor='roc_auc_val',patience=200, verbose=2,mode='max')
]
model = Sequential()
model.add(keras.layers.core.Masking(mask_value=0., input_shape=(max_lenth, max_features)))
model.add(GRU(units=n_hidden_units,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,#多层时需设置为true
              return_state=False, go_backwards=False, stateful=False, unroll=False))   #input_shape=(max_lenth, max_features),
model.add(GRU(units=n_hidden_units,activation='selu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=False,
              return_state=False, go_backwards=False, stateful=False, unroll=False))   #input_shape=(max_lenth, max_features),
model.add(Dropout(0.5))
 
model.add(Dense(1))
model.add(BatchNormalization())
model.add(keras.layers.core.Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[metrics.binary_crossentropy])  
model.fit(x_train, y_train, batch_size=train_batch_size, epochs=training_iters, verbose=2,
          callbacks=cb,validation_split=0.2,
shuffle=True, class_weight=class_weight, sample_weight=None, initial_epoch=0)
pred_y = model.predict(x_test, batch_size=test_batch_size)
score = roc_auc_score(y_test,pred_y)
