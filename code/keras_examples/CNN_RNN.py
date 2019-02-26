# CNN-RNN融合


class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)
 
    def build(self, input_shape):
        input_shape = input_shape
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        return x
 
    def get_output_shape_for(self, input_shape):
        return input_shape
 
model_left = Sequential()
model_left.add(keras.layers.core.Masking(mask_value=0., input_shape=(max_lenth, max_features)))  #解决不同长度的序列问题
model_left.add(GRU(units=left_hidden_units,activation='relu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,#多层时需设置为true
              return_state=False, go_backwards=False, stateful=False, unroll=False))
model_left.add(GRU(units=left_hidden_units,activation='relu',kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
              bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),
              bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
              bias_constraint=None, dropout=0.5, recurrent_dropout=0.0, implementation=1, return_sequences=True,
              return_state=False, go_backwards=False, stateful=False, unroll=False))
model_left.add(NonMasking())   #Flatten()不支持masking,此处用于unmask
model_left.add(Flatten())
 
## FCN
model_right = Sequential()
model_right.add(Conv1D(128, 3, padding='same', input_shape=(max_lenth, max_features)))
model_right.add(BatchNormalization())
model_right.add(Activation('relu'))
model_right.add(Conv1D(256, 3))
model_right.add(BatchNormalization())
model_right.add(Activation('relu'))
model_right.add(Conv1D(128, 3))
model_right.add(BatchNormalization())
model_right.add(Activation('relu'))
model_right.add(GlobalAveragePooling1D())
model_right.add(Reshape((1,1,-1)))
model_right.add(Flatten())
 
model = Sequential()
model.add(Merge([model_left,model_right], mode='concat'))
 
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
model.fit([left_x_train,right_x_train], y_train, batch_size=train_batch_size, epochs=training_iters, verbose=2,
          callbacks=[cb],validation_split=0.2,
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
pred_y = model.predict([left_x_test,right_x_test], batch_size=test_batch_size)
score = roc_auc_score(y_test,pred_y)

