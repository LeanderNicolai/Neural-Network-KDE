from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.utils import to_categorical


(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain_rs = xtrain.reshape(60000, 28, 28, 1)
xtest_rs = xtest.reshape(10000, 28, 28, 1)
y_train_hot = pd.get_dummies(ytrain)
y_train_hot = y_train_hot.values
y_test_hot = pd.get_dummies(ytest)
y_test_hot = y_test_hot.values

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(xtrain_rs, y_train_hot, validation_data=(xtest_rs, y_test_hot), epochs=3)

model.save_weights('models/CNN_1.h5')
