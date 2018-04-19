from preprocessing.data import load_data
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras import optimizers


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/data'

def build_model(input_size, output_size):
    model = Sequential()

    #heads up, it expects 4D input (batch, rows, cols, channels) - this can be changed if we change data_format
    model.add(ZeroPadding3D(padding=(3,2,2), name='padding1', input_shape=(input_size)))
    #not too sure why they use he initialization https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are
    #Rather than xavier (I used xavier initialization for S&G)
    model.add(Conv3D(32, (7,5,5), strides=(3,2,2), activation='relu', kernel_initializer='glorot_normal', name='conv1'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='max1'))
    model.add(Dropout(0.5))    

    model.add(ZeroPadding3D(padding=(2,2,2), name='padding2') )
    #not entirely sure whether striding in time domain is a good idea
    model.add(Conv3D(32, (5,5,5), strides=(2,2,2), activation='relu', kernel_initializer='glorot_normal', name='conv2'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='max2'))
    model.add(Dropout(0.5))  

    model.add(ZeroPadding3D(padding=(1,2,2), name='padding3') )
    model.add(Conv3D(32, (3,5,5), strides=(1,2,2), activation='relu', kernel_initializer='glorot_normal', name='conv3'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='max3'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding3D(padding=(1,2,2), name='padding4') )
    model.add(Conv3D(64, (3,5,5), strides=(1,1,1), activation='relu', kernel_initializer='glorot_normal', name='conv4'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='max4'))
    model.add(Dropout(0.5))

    model.add(ZeroPadding3D(padding=(1,2,2), name='padding5') )
    model.add(Conv3D(96, (3,3,3), strides=(1,1,1), activation='relu', kernel_initializer='glorot_normal', name='conv5'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='max5'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    #consider adding regularizer here if we're overfitting 
    model.add(Dense(output_size, activation='softmax', kernel_initializer='glorot_normal'))

    #slow decay
    adam = optimizers.Adam(lr=.001, decay=.001)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def train(model, x_train, y_train, batch_size=256, epochs=100, val_train_ratio=0.2):
    history = model.fit(x_train, y_train,
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
    print("loading images...")
    for i, (img, word) in enumerate(load_data(DATA_PATH, verbose=True, framebyframe=False)):
        x.append(img)
        y.append(word)
        if i == 5:
            break

    print("convering to np array...")
    x = np.stack(x, axis=0)

    print("transforming y...")
    y = le.fit_transform(y)
    y = oh.fit_transform(y.reshape(-1, 1)).todense()

    return x, y

def main():
    x, y = read_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = build_model(x.shape[1:], y.shape[1:])
    history = train(model, x_train, y_train)

    print("Saving model...")
    model.model.save('model.h5') 

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