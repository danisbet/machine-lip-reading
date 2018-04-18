from preprocessing.data import load_data
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/data'

def build_model():
    model = Sequential()

    # Build model here...

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
    for i, (img, word) in enumerate(load_data(DATA_PATH, verbose=True, framebyframe=True)):
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
    epochs = 10
    x, y = read_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = build_model()
    history = train(model, x_train, y_train, epochs=epochs)

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
