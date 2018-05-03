from preprocessing.data_lstm import read_data_for_speaker
from preprocessing.data_lstm import get_sil_image
from lstm_utils.callbacks import Statistics
import matplotlib.pyplot as plt
import os
import numpy as np

import time
import datetime
import argparse

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import GRU, LSTM
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding3D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras import backend as K

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/data'



def build_model(input_size, output_size=28, max_string_len=10):
    #########################################################
    # Model: conv2d + conv2d + maxpool + dropout + conv2d +
    #  bidirectional gru + dense + softmax + ctc loss
    #########################################################

    ## input_size: placeholder in Keras
    #  shape: (None, seq_size = 20, height = 50, width = 100, channels = 3)
    # if K.image_data_format() == 'channels_first':
    #     input_size = (self.img_c, self.frames_n, self.img_w, self.img_h)
    # else:
    #     input_size = (self.frames_n, self.img_w, self.img_h, self.img_c)

    # self.input_data = Input(name='the_input', shape=input_size, dtype='float32')
    input_data = Input(name='the_input', shape=input_size, dtype='float32')
    '''
    x = ZeroPadding3D(padding=(3, 5, 5), name='padding1')(input_data)
    x = Conv3D(filters=32, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation='relu',
           kernel_initializer='he_normal', name='conv1')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(x)
    x = Dropout(0.5)(x)

    x = ZeroPadding3D(padding=(1, 2, 2), name='padding2')(x)
    x = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='glorot_normal', name='conv2')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(x)
    x = Dropout(0.5)(x)

    x = ZeroPadding3D(padding=(1, 1, 1), name='padding3')(x)
    x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), activation='relu', kernel_initializer='glorot_normal', name='conv3')(x)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='max3')(x)
    x = Dropout(0.5)(x)
    '''
    ## padding used on the height and width before convolving
    #  shape:(None, 20, 54, 104, 3)
    model = Sequential()
    model.add(ZeroPadding3D(padding=(0, 2, 2), name='padding1'))

    ## 2D Convolution on each time sequence, relu activation
    #  shape 1st conv: (None, 20, 27, 52, 32)
    #  shape 2nd conv: (None, 20, 14, 26, 32)
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=5, kernel_initializer='he_normal', strides=(2, 2),
                               padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=5, kernel_initializer='he_normal', strides=(2, 2),
                               padding='same', activation='relu')))

    ## Max pool on each time sequence and Dropout
    #  shape maxpool: (None, 20, 7, 13, 32)
    #  shape dropout: (None, 20, 7, 13, 32)
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=None, name='max1')))
    model.add(Dropout(0))

    ## 2D Convolution on each time sequence, relu activation
    #  shape 1st conv: (None, 20, 4, 7, 4)
    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=5, kernel_initializer='he_normal', strides=(2, 2),
                               padding='same', activation='relu')))

    ## Flatten to gru
    #  shape: (None, 20, 112)
    model.add(TimeDistributed(Flatten()))

    ## Bidirectional gru
    #  shape: (None, 20, 512)
    # x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm1')(input_lstm)
    # x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2')(x_lstm)
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
                           merge_mode='concat'))
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
                           merge_mode='concat'))
    ## dense (512, 28) with softmax
    model.add(Flatten())
    #  shape: (None, 20, 28)
    model.add(Dense(128, kernel_initializer='he_normal', name='dense1'))
    model.add(Dense(output_size, kernel_initializer='he_normal', name='dense2'))

    ## prepare input for ctc loss
    model.add(Activation('softmax', name='softmax'))
    #labels = Input(name='the_labels', shape=[max_string_len], dtype='int32')
    #input_length = Input(name='input_length', shape=[1], dtype='int32')
    #label_length = Input(name='label_length', shape=[1], dtype='int32')
    #loss = CTC('ctc', [y_pred, labels, input_length, label_length])
    model.summary()

    return model


def pad_labels(labels, max_string_len=10):
    ################################################################
    # This method uses for padding labels, default string len is 10
    # return numpy array of padded input of shape (None, 20)
    ################################################################
    padding = np.ones((labels.shape[0], max_string_len - labels.shape[1])) * -1
    return np.concatenate((labels, padding), axis=1)


def train(model, x_train, y_train, batch_size=256, epochs=100, val_train_ratio=0.2,
          start_epoch=0):
    ##
    # Train model, typically will train for each speaker
    ## padding the labels
    max_string_len = 10
    if y_train.shape[1] != max_string_len:
        y_train = pad_labels(y_train, max_string_len)

    adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    if start_epoch > 0:
        weight_file = os.path.join(CURRENT_PATH, "model_lstm_oh.h5")
        model.load_weights(weight_file)
    ## callbacks when each epoch ends
    #  This will ouput character error rate which
    #  compares each predicted word with source word.
    #  TODO: results file need to be implemented
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_train_ratio,
                        shuffle=True,
                        initial_epoch=start_epoch,
                        verbose=1)

    return history


def read_data():
    ###############################
    # only works for CNN (not test)
    ###############################
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


def pad_input(x, max_str_len):
    #################
    # pad on axis = 1
    #################
    # sil_image = get_sil_image()
    # sil_image = [sil_image for _ in range(max_str_len - x.shape[1])]
    # sil_image = np.stack(sil_image, axis=0)
    # padding = [sil_image for _ in range(x.shape[0])]
    # padding = np.stack(padding, axis=0)
    padding = np.zeros((x.shape[0],max_str_len - x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
    return np.concatenate((x, padding), axis=1)


def main():
    ## add parser to initialize  start_epoch as well as learning rate
    parser = argparse.ArgumentParser(description=' CNN+GRU model for lip reading.')
    # parser.add_argument("-lr", default=0.00001, type=float, help="learning rate")
    parser.add_argument("-se", dest='start_epoch', default=0, type=int, help="start_epoch")
    parser.add_argument("-sid", dest='speaker_id', default=1, type=int, help="speaker id")
    args = parser.parse_args()
    start_epoch = args.start_epoch
    # start_epoch = 0
    speaker_id = args.speaker_id
    speaker_name = 's' + str(speaker_id)

    epochs = 200
    if start_epoch >= epochs:
        print "start_epoch too large, should be smaller than 2000!"

    max_seq_len = 25
    # x_s = np.ndarray(shape=(0, max_seq_len, 50, 100, 3))
    # y_s = np.ndarray(shape=(0, 6))
    # label_lens = np.array([])
    # input_lens = np.array([])

    start = time.time()
    # stack all the data
    # for count in range(1, 5):
    # TODO: this should be walk through files in np_s*
    # range(1,5) is number of data 'npz'
    # for count in range(1, 6):
    #     print("loading data for ", count)
    #     x, y = load_data(speaker_name, count)
    #     x = pad_input(x, max_seq_len)
    #     x_s = np.vstack((x, x_s))
    #     y_s = np.vstack((y, y_s))
    #     label_lens = np.concatenate([label_len, label_lens])
    #     input_lens = np.concatenate([input_len, input_lens])
    # TODO: add data path
    x_path = DATA_PATH + '/../X.npz'
    y_path = DATA_PATH + '/../y.npz'
    x_s = np.load(x_path)['x']
    y_s = np.load(y_path)['y']
    x_s = pad_input(x_s, max_seq_len)
    end = time.time()
    print("load data took", end - start)

    print(x_s.shape, y_s.shape)
    y_s = y_s.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x_s, y_s, test_size=0.2, shuffle=False)

    # 28 is outout size
    # run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    print x_train.shape
    model = build_model(x.shape[1:], 51, max_string_len=10)

    # input_len_train = np.ones((x_train.shape[0],1),dtype = np.int32)*max_seq_len
    history = train(model, x_train, y_train, batch_size=80, epochs=epochs,
                    start_epoch=start_epoch)

    print("Finish Training...")
    model.save('model_lstm_oh.h5')

    # TODO: add visualization
    print("Plotting...")
    ax2.plot(range(1, epochs + 1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, epochs + 1), history.history['val_loss'], 'tab:green', label="validation loss")
    f, (ax1, ax2) = plt.plot()
    ax1.plot(range(1, epochs+1), history.history['val_acc'], 'tab:blue', label="validation accuracy")
    ax1.plot(range(1, epochs+1), history.history['acc'], 'tab:red', label="training accuracy")
    ax2.plot(range(1, epochs + 1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, epochs + 1), history.history['val_loss'], 'tab:green', label="validation loss")

    ax1.legend()
    ax2.legend()

    f.savefig('training.png', dpi=300)
    print("Done.")


if __name__ == "__main__":
    main()
