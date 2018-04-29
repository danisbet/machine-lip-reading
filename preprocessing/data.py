import numpy as np
import os
from align import read_align
from video import read_video
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/../data'
PREDICTOR_PATH = CURRENT_PATH + '/shape_predictor_68_face_landmarks.dat'


def text_to_labels(text):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret

def labels_to_text(labels):
# 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text

def load_data(datapath, verbose=False, num_samples=-1, ctc_encoding=False):
    oh = OneHotEncoder()
    le = LabelEncoder()

    counter = 0
    done = False

    max_len = 0
    max_word_len = 0

    x = list()
    y = list()
    for root, dirs, files in os.walk(datapath):
        for name in files:
            if '.mpg' in name:
                if verbose is True:
                    print("reading: " + root + name)

                video = read_video(os.path.join(root, name), PREDICTOR_PATH)
                alignments = read_align(os.path.join(root, '../align/', name.split(".")[0] + ".align"))
                
                if video.shape[1] > 50 or video.shape[2] > 100 or video.shape[3] > 3:
                    continue

                for start, stop, word in alignments:
                    if word == 'sil' or word == 'sp':
                        continue
                   
                    if start < stop and stop < len(video):
                        x.append(video[start:stop])
                        y.append(word)
                    else:
                        continue

                    max_word_len = max(max_word_len, len(word))
                    max_len = max(max_len, stop-start)

                    counter += 1
                    if counter == num_samples:
                        done = True
                        break
            if done:
                break
        if done:
            break
    
    if not ctc_encoding:
        print("Encoding y...")
        y = le.fit_transform(y)
        y = oh.fit_transform(y.reshape(-1, 1)).todense()

    print("Formatting X...")
    for i in range(len(x)):
        result = np.zeros((max_len, 50, 100, 3))
        result[:x[i].shape[0], :x[i].shape[1], :x[i].shape[2], :x[i].shape[3]] = x[i]
        x[i] = result

        if ctc_encoding:
            #print("Encoding y...")
            res = np.ones(max_word_len) * -1
            enc = np.array(text_to_labels(y[i]))
            res[:enc.shape[0]] = enc
            y[i] = res

    if ctc_encoding:
        print("Stacking y...")
        y = np.stack(y, axis=0)

    x = np.stack(x, axis=0)
    
    return x, y

if __name__ == "__main__":
    X, y = load_data(DATA_PATH, verbose=True, ctc_encoding=False, num_samples=-1)
    print("X:", X.shape)
    print("y:", y.shape)
    
    np.savez_compressed('X', x=X)
    np.savez_compressed('y', y=y)
