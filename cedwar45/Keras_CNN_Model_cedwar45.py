

#From https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py



from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from cedwar45.ConfusionMatrix import ConfusionMatrix


np.random.seed(123);

batch_size = 128
num_classes = 10
epochs = 30


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets

fashion = True;#set if you want fashion-mnist

if fashion:
    #import mnist_reader
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

else: #use regular MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

c = 10

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test_orig = y_test
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=4,
                              verbose=0, mode='auto')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

              
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          #verbose=1,
          callbacks = [cb],
          validation_split = .1,
          #validation_data=(X_test, y_test)
          )
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predicted = model.predict(X_test,verbose=False)
predicted = np.argmax(predicted, axis = 1);


CM = ConfusionMatrix(predicted, y_test_orig, c);
np.savetxt("data/CNN_predicted_raw.txt", predicted, "%d")
np.savetxt("data/CNN_cm_raw.txt", CM, "%d");




