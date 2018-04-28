from preprocessing.data import load_data
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential

from cnn import Cnn 

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/data'

def main():
    epochs = 10
    x, y = load_data(DATA_PATH, verbose=False, num_samples=5)

    print("training data shapes:", x.shape, y.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print("building model...")
    print(x.shape)
    print(y.shape[1])
    model = Cnn(x.shape[1:], y.shape[1])
    model.build()



    history = model.train(x_train, y_train, epochs=epochs)

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
