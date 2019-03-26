import gzip
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

codes = np.eye(10)
with gzip.open('mnist.pkl.gz', 'rb') as fd:
    data = pickle.loads(fd.read(), encoding='latin')
(train_x, train_y), (val_x, val_y), (test_x, test_y) = data
train_y, val_y, test_y = [codes[y] for y in [train_y, val_y, test_y]]

model = Sequential()
model.add(Dense(784, input_shape=(784,)))
model.add(Dense(100, input_shape=(784,), activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

optimizer = SGD(lr=0.05, momentum=0.8)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=16, epochs=1, validation_data=(val_x, val_y))
print(model.evaluate(test_x, test_y))


print(model.predict(np.array([test_x[0]]), batch_size=1))
