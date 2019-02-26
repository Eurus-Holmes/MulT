# CNN处理时序数据的二分类



model = Sequential()
model.add(Conv1D(128, 3, padding='same', input_shape=(max_lenth, max_features)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(256, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv1D(128, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(GlobalAveragePooling1D())   #时序的时间维度上全局池化
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[metrics.binary_crossentropy]) 
