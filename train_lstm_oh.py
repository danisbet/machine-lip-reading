from lstm_utils.callbacks import Statistics
from preprocessing.data_lstm import read_data_for_speaker
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
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/global/scratch/alex_vlissidis/lipreading_data"


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
    input_data = Input(name='input', shape=input_size, dtype='float32')
    
    ## padding used on the height and width before convolving
    #  shape:(None, 20, 54, 104, 3)
    model = Sequential()
    model.add(ZeroPadding3D(padding=(0, 2, 2),input_shape=(input_size), name='padding1', data_format='channels_last'))
    #model.add(TimeDistributed(Conv2D(filters=34, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', activation='relu')))

    model.add(TimeDistributed(Conv2D(filters=42, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max1')))

    model.add(TimeDistributed(Conv2D(filters=96, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(1, 1), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max2')))

    model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', activation='relu')))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max2')))

    ## Flatten to gru
    #  shape: (None, 20, 112)
    model.add(TimeDistributed(Flatten()))

    ## Bidirectional gru
    #  shape: (None, 20, 512)
    # x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm1')(input_lstm)
    # x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2')(x_lstm)
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat'))
    model.add(Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat'))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer='he_normal', name='fc1', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.7))
    model.add(Dense(output_size, kernel_initializer='he_normal', name='dense2'))

    ## prepare input for ctc loss
    model.add(Activation('softmax', name='softmax'))
    #loss = CTC('ctc', [y_pred, labels, input_length, label_length])
    model.summary()

    return model

def build_model_vgg16(input_size, output_size=28):
    K.set_image_data_format('channels_last')
    input_data = Input(name='input', shape=input_size, dtype='float32')

    # VGG16
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_size[1:], pooling='max')
    x = TimeDistributed(vgg16)(input_data)
    x = TimeDistributed(Flatten())(x)

    # Bi-GRU
    x = Bidirectional(GRU(256, return_sequences=True, name='gru1'), merge_mode='concat')(x)
    x = Bidirectional(GRU(256, return_sequences=True, name='gru1'), merge_mode='concat')(x)
    x = Flatten()(x)

    # FC + softmax
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(output_size, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=input_data, outputs=predictions)
    model.summary()

    # freeze all convolutional VGG16 layers
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


def train(model, x_train, y_train, batch_size=128, epochs=100, val_train_ratio=0.2, start_epoch=0):
    ##
    # Train model, typically will train for each speaker
    ## padding the labels
    # max_string_len = 10
    # if y_train.shape[1] != max_string_len:
    #     y_train = pad_labels(y_train, max_string_len)

    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.005, patience=10, verbose=1, mode='max')

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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
                        verbose=1,
                        callbacks=[early_stopping])
    return history


def pad_input(x, max_str_len):
    #################
    # pad on axis = 
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

    epochs = 100
    if start_epoch >= epochs:
        print("start_epoch too large, should be smaller than 2000!")

    max_seq_len = 40
   
    
    start = time.time()
   
    x = np.load('/global/scratch/alex_vlissidis/lipreading_data/s' + str(speaker_id) + '/X.npz')['x']
    y = np.load('/global/scratch/alex_vlissidis/lipreading_data/s' + str(speaker_id) + '/y.npz')['y']
    print(x.shape, y.shape)
    y = y.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
 
    end = time.time()
    print("load data took", end-start)

    # 28 is outout size
    # run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    print(x_train.shape)
    model = build_model_vgg16(x_train.shape[1:], y_train.shape[1])

    # input_len_train = np.ones((x_train.shape[0],1),dtype = np.int32)*max_seq_len
    history = train(model, x_train, y_train, batch_size=256, epochs=epochs, start_epoch=start_epoch)

    print("Finish Training...")
    model.save('/global/scratch/alex_vlissidis/lipreading_models/model_gru-fc4096-lr0.0001-dr0.2-'+ speaker_name + '.h5')

    print("Plotting...")
    nepochs = len(history.history['acc'])
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(1, nepochs+1), history.history['val_acc'], 'tab:blue', label="validation accuracy")
    ax1.plot(range(1, nepochs+1), history.history['acc'], 'tab:red', label="training accuracy")

    ax2.plot(range(1, nepochs+1), history.history['loss'], 'tab:orange', label="loss")
    ax2.plot(range(1, nepochs+1), history.history['val_loss'], 'tab:green', label="validation loss")

    ax1.legend()
    ax2.legend()
    f.savefig('figures/training_vgg16-gru-fc4096-lr0.0001-' + speaker_name + '.png', dpi=300)

    print("Testing:", model.evaluate(x=x_test, y=y_test, batch_size=32))
    print("Done.")

if __name__ == "__main__":
    main()
