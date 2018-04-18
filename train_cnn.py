from preprocessing.data import load_data
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/data'

def build_model():
    pass

def train():
    pass

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
    x, y = read_data()

    print(x.shape, y.shape)

    #model = build_model()
    #train(model)

if __name__ == "__main__":
    main()
