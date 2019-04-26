from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain_rs = xtrain.reshape(60000,784)
xtest_rs = xtest.reshape(10000,784)
y_train_hot = pd.get_dummies(ytrain)
y_train_hot = y_train_hot.values
y_test_hot = pd.get_dummies(ytest)
y_test_hot = y_test_hot.values

model = Sequential([
    Dense(100, input_shape=(784,)),
    Activation('softmax'),
    Dense(10),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(xtrain_rs, y_train_hot, epochs=500, batch_size=300)

train_score = model.evaluate(xtrain_rs, y_train_hot, batch_size=300)
print('training score is: ', train_score)

test_score = model.evaluate(xtest_rs, y_test_hot, batch_size=300)

print('test score is: ', test_score)

test_pred_hot = model.predict(xtest_rs)

# test_pred = [np.where(r==1)[0][0] for r in test_pred_hot]
a = test_pred_hot
a = (a == a.max(axis=1)[:,None]).astype(int)
test_pred_nums = [np.where(r==1)[0][0] for r in a]

pred_array = np.array(test_pred_nums)

print('pred array shape is: ', pred_array.shape)
print('ytest array shape is: ', pred_array.shape)


print(model.summary())
