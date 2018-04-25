import numpy as np
import os
from align import read_align
from video import read_video
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/../data'
PREDICTOR_PATH = CURRENT_PATH + '/shape_predictor_68_face_landmarks.dat'


def load_data(datapath, verbose=False, num_samples=5):
    oh = OneHotEncoder()
    le = LabelEncoder()

    counter = 0
    done = False
    max_len = 0

    x_raw = list()
    y_raw = list()
    for root, dirs, files in os.walk(datapath):
        for name in files:
            if '.mpg' in name:
                if verbose is True:
                    print("reading: " + root + name)

                video = read_video(os.path.join(root, name), PREDICTOR_PATH)
                alignments = read_align(os.path.join(root, '../align/', name.split(".")[0] + ".align"))
                
                for start, stop, word in alignments:
                    x_raw.append(video[start:stop])
                    y_raw.append(word)

                    max_len = max(max_len, stop-start)

                    counter += 1
                    if counter == num_samples:
                        done = True
                        break
            if done:
                break
        if done:
            break
                
    y_raw = le.fit_transform(y_raw)
    y = oh.fit_transform(y_raw.reshape(-1, 1)).todense()

    for i in range(len(x_raw)):
        result = np.zeros((max_len, x_raw[i].shape[1], x_raw[i].shape[2], x_raw[i].shape[3]))
        result[:x_raw[i].shape[0], :x_raw[i].shape[1], :x_raw[i].shape[2], :x_raw[i].shape[3]] = x_raw[i]
        x_raw[i] = result

    x = np.stack(x_raw, axis=0)
    
    return x, y

if __name__ == "__main__":
    X, y = load_data(DATA_PATH, verbose=True)
    print("X:", X.shape)
    print("y:", y.shape)
