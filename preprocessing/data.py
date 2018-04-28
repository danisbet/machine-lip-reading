import numpy as np
import os
from align import read_align
from video import read_video
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import re

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

    word_len_list = []
    input_len_list = []

    x_raw = list()
    y_raw = list()
    pattern = re.compile("s[0-9]")
    speakers = []
    
    for root, dirs, files in os.walk(datapath):
        check = root.split("/")[-1]
        match = pattern.findall(check)
        if (len(match) > 0):
            if check.index(match[0]) == 0:
                if verbose:
                    print(check, match[0])
                speakers.append(check)
        
        for name in files:
            if '.mpg' in name:
                if verbose is True:
                    print("reading: " + root + "/" + name)

                video = read_video(os.path.join(root, name), PREDICTOR_PATH)
                alignments = read_align(os.path.join(root, '../align/', name.split(".")[0] + ".align"))
                
                for start, stop, word in alignments:
                    if word == 'sil' or word == 'sp':
                        continue
                    
                    if verbose is True:
                        print(str(counter) + ": " + str(start) + "--" + str(stop) + ": " + word)
                    
                    _, d1, d2, d3 = video[start:stop].shape
                    
                    if (len(x_raw) > 0):
                        _, prev_d1, prev_d2, prev_d3 = x_raw[-1].shape
                        if (d1, d2, d3) != (prev_d1, prev_d2, prev_d3):
                            if verbose is True:
                                print("different size, skip")
                            continue
                    
                    x_raw.append(video[start:stop])
                    y_raw.append(word)

                    max_word_len = max(max_word_len, len(word))
                    max_len = max(max_len, stop-start)
                    word_len_list.append(len(word))
                    input_len_list.append(stop-start)
                    counter += 1
                    if counter == num_samples:
                        done = True
                        break
            
            if done:
                break
        
        if done:
            break
    
    if not ctc_encoding:
        y_raw = le.fit_transform(y_raw)
        y = oh.fit_transform(y_raw.reshape(-1, 1)).todense()


    for i in range(len(x_raw)):
        result = np.zeros((max_len, x_raw[i].shape[1], x_raw[i].shape[2], x_raw[i].shape[3]))
        result[:x_raw[i].shape[0], :x_raw[i].shape[1], :x_raw[i].shape[2], :x_raw[i].shape[3]] = x_raw[i]
        if verbose is True:
            print(str(i) + ": " + str(result.shape))
        x_raw[i] = result

        if verbose is True:
            print("Added: " + str(x_raw[i].shape)) 
        
        if ctc_encoding:
            res = np.ones(max_word_len) * -1
            enc = np.array(text_to_labels(y_raw[i]))
            res[:enc.shape[0]] = enc
            y_raw[i] = res

    if ctc_encoding:
        y = np.stack(y_raw, axis=0)

    x = np.stack(x_raw, axis=0)
#     print(root)
#     np.save("test_savex", x_raw)
#     np.save("test_savey", y_raw)
    return x, y, np.array(word_len_list), np.array(input_len_list)

if __name__ == "__main__":
    X, y = load_data(DATA_PATH, verbose=True, ctc_encoding=True, num_samples=15)
    np.savetext("test_save", (X, y))
    print("X:", X.shape)
    print("y:", y.shape)
    print(y)
