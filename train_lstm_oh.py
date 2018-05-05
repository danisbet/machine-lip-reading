from lstm_utils.callbacks import Statistics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

import time
import datetime
import argparse

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import GRU, LSTM
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding3D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import regularizers

from sklearn.model_selection import train_test_split

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/global/scratch/alex_vlissidis"


def build_model(input_size, output_size=28):
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
    
    ## padding used on the height and width before convolving
    #  shape:(None, 20, 54, 104, 3)
    model = Sequential()
    model.add(ZeroPadding3D(padding=(0, 2, 2),input_shape = (input_size), name='padding1'))

    ## 2D Convolution on each time sequence, relu activation
    #  shape 1st conv: (None, 20, 27, 52, 32)
    #  shape 2nd conv: (None, 20, 14, 26, 32)
    model.add(TimeDistributed(Conv2D(filters=34, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max1')))

    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max2')))
    model.add(Dropout(0))

    ## 2D Convolution on each time sequence, relu activation
    #  shape 1st conv: (None, 20, 4, 7, 4)
    model.add(TimeDistributed(Conv2D(filters=96, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', activation='relu')))

    ## Flatten to gru
    #  shape: (None, 20, 112)
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.0))
    ## Bidirectional gru
    #  shape: (None, 20, 512)
    # x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm1')(input_lstm)
    # x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2')(x_lstm)
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'),
                           merge_mode='concat'))
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'),
                           merge_mode='concat'))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_normal', name='fc1', kernel_regularizer=regularizers.l2(0.0)))
    model.add(Dense(output_size, kernel_initializer='he_normal', name='dense2'))

    ## prepare input for ctc loss
    model.add(Activation('softmax', name='softmax'))
    #loss = CTC('ctc', [y_pred, labels, input_length, label_length])
    model.summary()

    return model

def build_model_vgg16(input_size, output_size=28):
    input_data = Input(name='the_input', shape=input_size, dtype='float32')

    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_size[1:], pooling='max')
    x = TimeDistributed(vgg16)(input_data)
    x = TimeDistributed(Flatten())(x)

    # let's add a fully-connected layer
    x = Bidirectional(GRU(256, return_sequences=True, name='gru1'), merge_mode='concat')(x)
    x = Bidirectional(GRU(256, return_sequences=True, name='gru1'), merge_mode='concat')(x)
    x = Flatten()(x)

    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(output_size, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=input_data, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in vgg16.layers:
       layer.trainable = False

    return model

def pad_labels(labels, max_string_len=10):
    ################################################################
    # This method uses for padding labels, default string len is 10
    # return numpy array of padded input of shape (None, 20)
    ################################################################
    padding = np.ones((labels.shape[0], max_string_len - labels.shape[1])) * -1
    return np.concatenate((labels, padding), axis=1)


def train(model, x_train, y_train, batch_size=256, epochs=100, val_train_ratio=0.2, start_epoch=0):
    ##
    # Train model, typically will train for each speaker
    ## padding the labels
    # max_string_len = 10
    # if y_train.shape[1] != max_string_len:
    #     y_train = pad_labels(y_train, max_string_len)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    if start_epoch > 0:
        weight_file = os.path.join(CURRENT_PATH, "models/model_lstm_oh.h5")
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

    epochs = 20
    if start_epoch >= epochs:
        print("start_epoch too large, should be smaller than 2000!")

    max_seq_len = 40
    # x_s = np.ndarray(shape=(0, max_seq_len, 50, 100, 3))
    # y_s = np.ndarray(shape=(0, 6))
    # label_lens = np.array([])
    # input_lens = np.array([])

    start = time.time()
    
    x_path = DATA_PATH + '/X.npz'
    y_path = DATA_PATH + '/y.npz'
    x_s = np.load(x_path)['x']
    y_s = np.load(y_path)['y']
    #x_s = pad_input(x_s, max_seq_len)
    
    end = time.time()
    print("load data took", end - start)

    print(x_s.shape, y_s.shape)
    y_s = y_s.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x_s, y_s, test_size=0.2, shuffle=True)

    # 28 is outout size
    # run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    print(x_train.shape)
    model = build_model_vgg16(x_train.shape[1:], y_train.shape[1])

    # input_len_train = np.ones((x_train.shape[0],1),dtype = np.int32)*max_seq_len
    history = train(model, x_train, y_train, batch_size=256, epochs=epochs,
                    start_epoch=start_epoch)

    print("Finish Training...")
    model.save('models/model_lstm_oh.h5')

    print("Plotting...")
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(1, epochs+1), history.history['val_acc'], 'tab:blue', label="validation accuracy")
    ax1.plot(range(1, epochs+1), history.history['acc'], 'tab:red', label="training accuracy")

    ax2.plot(range(1, epochs+1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, epochs+1), history.history['val_loss'], 'tab:green', label="validation loss")

    ax1.legend()
    ax2.legend()
    f.savefig('figures/training_lstm.png', dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()
