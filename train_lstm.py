from preprocessing.data import load_data
import matplotlib.pyplot as plt
import os
import numpy as np

import time

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import GRU
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding3D
from keras.layers.core import Lambda, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras import backend as K

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/data'

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def CTC(name, args):
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)


def build_model(input_size, output_size = 28, max_string_len = 10):
    # model = Sequential()
    input_data = Input(name='the_input', shape=input_size, dtype='float32')
    x = ZeroPadding3D(padding=(0,2,2), name='padding1')(input_data)
    x = TimeDistributed(Conv2D(filters = 32, kernel_size = 5, strides = (2,2),
                             padding = 'same', activation = 'relu'))(x)
    print
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=None, name='max1'))(x)
    x = Dropout(0.5)(x)

    x = TimeDistributed(Conv2D(filters=32, kernel_size=5, strides=(2, 2),
                               padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=None, name='max1'))(x)
    x = Dropout(0.5)(x)

    x = TimeDistributed(Conv2D(filters=4, kernel_size=5, strides=(2, 2),
                               padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=None, name='max1'))(x)
    x = Dropout(0.5)(x)

    input_lstm = TimeDistributed(Flatten())(x)

    x_lstm = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(input_lstm)
    x_lstm = Dense(output_size, kernel_initializer='he_normal', name='dense1')(x_lstm)
    print "after dense1"
    y_pred = Activation('softmax', name='softmax')(x_lstm)

    labels = Input(name='the_labels', shape = [max_string_len], dtype='float32')
    input_length = Input(name = 'input_length', shape =[1], dtype = 'int64')
    label_length = Input(name = 'label_length', shape = [1], dtype = 'int64')
    loss = CTC('ctc',[y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_data, labels, label_length, input_length],
                  outputs = loss)
    model.summary()
    # Build model here...

    return model
def pad_labels(labels, max_string_len):
    padding = np.ones((labels.shape[0], max_string_len - labels.shape[1])) * -1
    return np.concatenate((labels, padding), axis = 1)

def train(model, x_train, y_train, label_len_train, input_len_train, batch_size=256, epochs=100, val_train_ratio=0.2):
    max_string_len = 10
    if y_train.shape[1] != max_string_len:
        y_train = pad_labels(y_train, max_string_len)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    history = model.fit(x = {'the_input':x_train, 'the_labels':y_train, 'label_length':label_len_train,
                             'input_length':input_len_train}, y = {'ctc': np.zeros([x_train.shape[0]])},
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_train_ratio,
                        shuffle=True,
                        verbose=1)

    return history

def read_data():
    oh = OneHotEncoder()
    le = LabelEncoder()

    x = list()
    y = list()
    t = list()
    print("loading images...")
    for i, (img, words) in enumerate(load_data(DATA_PATH, verbose=False, framebyframe=False)):
        if img.shape[0] != 75:
            continue
        x.append(img)
        y.append(words)

        t += words.tolist()
        if i == 3:
            break

    t = le.fit_transform(t)
    oh.fit(t.reshape(-1, 1))

    print("convering to np array...")
    x = np.stack(x, axis=0)

    print("transforming y...")
    for i in range(len(y)):
        y_ = le.transform(y[i])
        y[i] = np.asarray(oh.transform(y_.reshape(-1, 1)).todense())
    y = np.stack(y, axis=0)

    return x, y

def main():
    epochs = 10

    start = time.time()
    print("loading data")
    x, y, label_len, input_len= load_data(DATA_PATH, verbose=False, num_samples=100, ctc_encoding=True)
    end = time.time()

    print("load data took", end-start)
    print("training data shapes:", x.shape, y.shape)
    x_train, x_test, y_train, y_test, label_len_train, label_len_test, \
    input_len_train, input_len_test = train_test_split(x, y, label_len, input_len, test_size=0.2)

    model = build_model(x.shape[1:], 28, max_string_len = 10)

    history = train(model, x_train, y_train, label_len_train, input_len_train, epochs=epochs)

    print("Saving model...")
    model.save('model.h5')

    print("Plotting...")
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(1, epochs+1), history.history['val_acc'], 'tab:blue', label="validation accuracy")
    ax1.plot(range(1, epochs+1), history.history['acc'], 'tab:red', label="training accuracy")

    ax2.plot(range(1, epochs+1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, epochs+1), history.history['val_loss'], 'tab:green', label="validation loss")

    ax1.legend()
    ax2.legend()

    f.savefig('training.png', dpi=300)
    print("Done.")


if __name__ == "__main__":
    main()
