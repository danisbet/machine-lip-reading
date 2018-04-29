from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Input, concatenate
from keras import optimizers

from keras.models import Model


from keras import backend as K

class Cnn(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def build(self):
        self.input_data = Input(name='input', shape=self.input_size, dtype='float32')

        """
        Head One: A sequence of convolutions with size 3 temporally. This is currently the same as what lipnet implemented
        
        """
        self.head1_zero1 = ZeroPadding3D(padding=(1,2,2), name='head1_zero1')(self.input_data)
        self.head1_conv1 = Conv3D(16, (3,5,5), strides=(1,2,2), activation='relu', kernel_initializer='glorot_normal', name='head1_conv1')(self.head1_zero1)
        self.head1_maxp1 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='head1_maxp1')(self.head1_conv1)
        self.head1_drop1 = Dropout(0.5)(self.head1_maxp1)

        self.head1_zero2 = ZeroPadding3D(padding=(1,2,2), name='head1_zero2')(self.head1_drop1)
        self.head1_conv2 = Conv3D(32, (3,5,5), strides=(1,1,1), activation='relu', kernel_initializer='glorot_normal', name='head1_conv2')(self.head1_zero2)
        self.head1_maxp2 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='head1_maxp2')(self.head1_conv2)
        self.head1_drop2 = Dropout(0.5)(self.head1_maxp2)

        self.head1_zero3 = ZeroPadding3D(padding=(1,1,1), name='head1_zero3')(self.head1_drop2)
        self.head1_conv3 = Conv3D(64, (3,3,3), strides=(1,1,1), activation='relu', kernel_initializer='glorot_normal', name='head1_conv3')(self.head1_zero3)
        self.head1_maxp3 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='head1_maxp3')(self.head1_conv3)
        self.head1_drop3 = Dropout(0.5)(self.head1_maxp3)

        self.head1_flat  = Flatten()(self.head1_drop3)

        print('head1 finished')

        """
        Head Two: Consider larger temporal features, 

        """
        self.head2_zero1 = ZeroPadding3D(padding=(3,2,2), name='head2_zero1')(self.input_data)
        self.head2_conv1 = Conv3D(4, (7,5,5), strides=(1,2,2), activation='relu', kernel_initializer='glorot_normal', name='head2_conv1')(self.head2_zero1)
        self.head2_maxp1 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='head2_maxp1')(self.head2_conv1)
        self.head2_drop1 = Dropout(0.5)(self.head2_maxp1)

        self.head2_zero2 = ZeroPadding3D(padding=(3,2,2), name='head2_zero2')(self.head2_drop1)
        self.head2_conv2 = Conv3D(16, (7,5,5), strides=(1,1,1), activation='relu', kernel_initializer='glorot_normal', name='head2_conv2')(self.head2_zero2)
        self.head2_maxp2 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='head2_maxp2')(self.head2_conv2)
        self.head2_drop2 = Dropout(0.5)(self.head2_maxp2)

        self.head2_zero3 = ZeroPadding3D(padding=(3,1,1), name='head2_zero3')(self.head2_drop2)
        self.head2_conv3 = Conv3D(32, (7,3,3), strides=(1,1,1), activation='relu', kernel_initializer='glorot_normal', name='head2_conv3')(self.head2_zero3)
        self.head2_maxp3 = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), name='head2_maxp3')(self.head2_conv3)
        self.head2_drop3 = Dropout(0.5)(self.head2_maxp3)

        self.head2_flat  = Flatten()(self.head1_drop3)
        print('head2 finished')


        """
        Combine heads and compile model
        """


        self.concat = concatenate([self.head1_flat, self.head2_flat], axis=1)
        temp_model = Model(input=self.input_data, output=self.concat)
        temp_model.summary()
        # self.flat = Flatten()(self.concat)
        self.predictions = Dense(self.output_size, activation='softmax', kernel_initializer='glorot_normal', name='dense1')(self.concat)
        
        model = Model(input=self.input_data, output=self.predictions)
        
        model.summary()
        
        adam = optimizers.Adadelta()
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print("model built")

        self.model = model

    def train(self, x_train, y_train, batch_size=256, epochs=100, val_train_ratio=0.2):
        print('training')
	history = self.model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,validation_split=val_train_ratio,shuffle=True,verbose=1)
        return history
