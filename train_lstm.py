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

from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import GRU,LSTM
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding3D, Conv3D, MaxPooling3D
from keras.layers.core import Lambda, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam
from keras import backend as K


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/data'

def ctc_lambda_func(args):
    ###################################################
    # wrap tf ctc loss function in Keras
    # In order to set ignore_longer_outputs_than_inputs
    # label need to import tensorflow
    ###################################################
    import tensorflow as tf
    y_pred, labels, input_length, label_length = args
    # From Keras example image_ocr.py:
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]

    label_length = K.cast(tf.squeeze(label_length),'int32')
    input_length = K.cast(tf.squeeze(input_length),'int32')
    labels = K.ctc_label_dense_to_sparse(labels, label_length)
    # y_pred = y_pred[:, :, :]
    # return K.ctc_batch_cost(labels, y_pred, input_length, label_length, ignore_longer_outputs_than_inputs=True)
    return tf.nn.ctc_loss(labels, y_pred, input_length, ctc_merge_repeated=False,
                         ignore_longer_outputs_than_inputs = True, time_major = False)


def CTC(name, args):
    ######################################
    # return CTC loss as a labmda function
    ######################################
	return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)


def build_model(input_size, output_size = 28, max_string_len = 10):
    #########################################################
    # Model: conv2d + conv2d + maxpool + dropout + conv2d +
    #  bidirectional gru + dense + softmax + ctc loss
    #########################################################

    ## input_size: placeholder in Keras
    #  shape: (None, seq_size = 20, height = 50, width = 100, channels = 3)
    input_data = Input(name='the_input', shape=input_size, dtype='float32')
    x = ZeroPadding3D(padding=(3, 2, 2), name='padding1', input_shape=(input_size))(input_data)
    x = Conv3D(filters=32, kernel_size=(7, 5, 5), strides=(1, 2, 2), padding='valid', activation='relu',
           kernel_initializer='glorot_normal', name='conv1')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(x)
    x = Dropout(0.5)(x)

    x = ZeroPadding3D(padding=(2, 2, 2), name='padding2')(x)
    x = Conv3D(32, (5, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='glorot_normal', name='conv2')(x)
    x = Dropout(0.5)(x)

    x = ZeroPadding3D(padding=(1, 2, 2), name='padding3')(x)
    x = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), activation='relu', kernel_initializer='glorot_normal', name='conv3')(x)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='max3')(x)
    x = Dropout(0.5)(x)
    '''
    ## padding used on the height and width before convolving
    #  shape:(None, 20, 54, 104, 3)
    x = ZeroPadding3D(padding=(0,2,2), name='padding1')(input_data)

    ## 2D Convolution on each time sequence, relu activation
    #  shape 1st conv: (None, 20, 27, 52, 32)
    #  shape 2nd conv: (None, 20, 14, 26, 32)
    x = TimeDistributed(Conv2D(filters = 64, kernel_size = 5, kernel_initializer='he_normal', strides = (2,2),
                             padding = 'same', activation = 'relu'))(x)
    x = TimeDistributed(Conv2D(filters=64, kernel_size=5, kernel_initializer='he_normal', strides=(2, 2),
                               padding='same', activation='relu'))(x)

    ## Max pool on each time sequence and Dropout
    #  shape maxpool: (None, 20, 7, 13, 32)
    #  shape dropout: (None, 20, 7, 13, 32)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2), strides=None, name='max1'))(x)
    x = Dropout(0.5)(x)

    ## 2D Convolution on each time sequence, relu activation
    #  shape 1st conv: (None, 20, 4, 7, 4)
    x = TimeDistributed(Conv2D(filters=32, kernel_size=5, kernel_initializer='he_normal', strides=(2, 2),
                               padding='same', activation='relu'))(x)
    '''
    ## Flatten to gru
    #  shape: (None, 20, 112)
    input_lstm = TimeDistributed(Flatten())(x)

    ## Bidirectional gru
    #  shape: (None, 20, 512)
    x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm1')(input_lstm)
    x_lstm = LSTM(256, return_sequences=True, kernel_initializer='Orthogonal', name='lstm2')(x_lstm)
    ## dense (512, 28) with softmax
    #  shape: (None, 20, 28)
    x_lstm = Dense(output_size, kernel_initializer='he_normal', name='dense1')(x_lstm)

    ## prepare input for ctc loss
    y_pred = Activation('softmax', name='softmax')(x_lstm)
    labels = Input(name='the_labels', shape = [max_string_len], dtype='int32')
    input_length = Input(name = 'input_length', shape =[1], dtype = 'int32')
    label_length = Input(name = 'label_length', shape = [1], dtype = 'int32')
    loss = CTC('ctc',[y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_data, labels, label_length, input_length],
                  outputs = loss)
    model.summary()

    return model

def pad_labels(labels, max_string_len = 10):
    ################################################################
    # This method uses for padding labels, default string len is 10
    # return numpy array of padded input of shape (None, 20)
    ################################################################
    padding = np.ones((labels.shape[0], max_string_len - labels.shape[1])) * -1
    return np.concatenate((labels, padding), axis = 1)

def train(model, x_train, y_train, label_len_train, input_len_train, batch_size=256, epochs=100, val_train_ratio=0.2, start_epoch = 0):

    ##
    # Train model, typically will train for each speaker
    ## padding the labels
    max_string_len = 10
    if y_train.shape[1] != max_string_len:
        y_train = pad_labels(y_train, max_string_len)

    adam = Adam(lr=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    if start_epoch > 0:
        weight_file = os.path.join(CURRENT_PATH, 'lstm_model/checkpoints', run_name , "weights{epoch:02d}.h5")
        model.load_weights(weight_file)
    ## callbacks when each epoch ends
    #  This will ouput character error rate which
    #  compares each predicted word with source word.
    #  TODO: results file need to be implemented
    stats = Statistics(model, x_train, y_train, input_len_train,
                        label_len_train, num_samples_stats=256, output_dir='lstm_model/results')
    ## TODO: add checkpoint
    # checkpoint = Checkpoint(os.path.join('lstm_model/checkpoints', "weights{epoch:02d}.h5"),
    #                         monitor='val_loss', save_weights_only=True, mode='auto', period=1)
    history = model.fit(x = {'the_input':x_train, 'the_labels':y_train, 'label_length':label_len_train,
                             'input_length':input_len_train}, y = {'ctc': np.zeros([x_train.shape[0]])},
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_train_ratio,
                        callbacks = [stats],
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
    sil_image = get_sil_image()
    sil_image = [sil_image for _ in range(max_str_len - x.shape[1])]
    sil_image = np.stack(sil_image, axis = 0)
    padding = [sil_image for _ in range(x.shape[0])]
    padding = np.stack(padding ,axis = 0)
    #padding = np.zeros((x.shape[0],max_str_len - x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
    return np.concatenate((x,padding),axis = 1)

def main():

    ## add parser to initialize  start_epoch as well as learning rate
    parser = argparse.ArgumentParser(description=' CNN+GRU model for lip reading.')
    #parser.add_argument("-lr", default=0.00001, type=float, help="learning rate")
    # parser.add_argument("-se", dest ='start_epoch', default=0, type=int, help="start_epoch")
    parser.add_argument("-sid", dest='speaker_id', default=1, type=int, help="speaker id")
    args = parser.parse_args()
    # start_epoch = args.start_epoch
    start_epoch = 0
    speaker_id = args.speaker_id
    speaker_name = 's'+str(speaker_id)

    epochs = 2000
    if start_epoch >= epochs:
        print "start_epoch too large, should be smaller than 2000!"

    max_seq_len = 25
    x_s = np.ndarray(shape=(0, max_seq_len, 50, 100, 3))
    y_s = np.ndarray(shape=(0, 6))
    label_lens = np.array([])
    input_lens = np.array([])

    start = time.time()
    # stack all the data
    #for count in range(1, 5):
    # TODO: this should be walk through files in np_s*
    # range(1,5) is number of data 'npz'
    for count in range(1, 6):
        print("loading data for ", count)
        x, y, label_len, input_len = read_data_for_speaker(speaker_name, count)
        x = pad_input(x, max_seq_len)
        x_s = np.vstack((x, x_s))
        y_s = np.vstack((y, y_s))
        label_lens = np.concatenate([label_len, label_lens])
        input_lens = np.concatenate([input_len, input_lens])
    end = time.time()
    print("load data took", end-start)

    print(x_s.shape, y_s.shape)
    x_train, x_test, y_train, y_test, label_len_train, label_len_test, \
    input_len_train, input_len_test = train_test_split(x_s, y_s, label_lens, input_lens, test_size=0.2)

    # 28 is outout size
    # run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model = build_model(x.shape[1:], 28, max_string_len=10)
    history = train(model, x_train, y_train, label_len_train, input_len_train, batch_size = 100, epochs=epochs, start_epoch = start_epoch)
    
    print("Finish Training...")
    model.save('model_lstm.h5')

    # TODO: add visualization
    # print("Plotting...")
    # ax2.plot(range(1, epochs + 1), history.history['loss'], 'tab:orange', label="loss")
    # ax2.plot(range(1, epochs + 1), history.history['val_loss'], 'tab:green', label="validation loss")
    #f, (ax1, ax2) = plt.subplots(2, 1)
    #ax1.plot(range(1, epochs+1), history.history['val_acc'], 'tab:blue', label="validation accuracy")
    #ax1.plot(range(1, epochs+1), history.history['acc'], 'tab:red', label="training accuracy")



    #ax1.legend()
    # ax2.legend()
    #
    # f.savefig('training.png', dpi=300)
    print("Done.")


if __name__ == "__main__":
    main()
