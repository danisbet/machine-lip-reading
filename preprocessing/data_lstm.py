import numpy as np
import os
from preprocessing.align import read_align
from preprocessing.video import read_video
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.ndimage import imread
# CURRENT_PATH = '/home/ubuntu/assignments/machine-lip-reading/preprocessing'
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

def get_sil_image():
    img_dir = os.path.join(CURRENT_PATH, 'sil_img.png')
    sil_img = imread(img_dir)
    return sil_img

def load_data(datapath, speaker, verbose=True, num_samples=1000, ctc_encoding=True):

    output_dir = "/global/scratch/alex_vlissidis/lipreading_data/" + speaker + '_np'
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    oh = OneHotEncoder()
    le = LabelEncoder()

    counter = 0
    done = False

    max_len = 0
    max_word_len = 0

    x = list()
    y = list()
    
    word_len_list = []
    input_len_list = []
    
    path = datapath + '/' + str(speaker)
    for root, dirs, files in os.walk(path):
        for name in files:
            if '.mpg' in name:
                if verbose is True:
                    print(str(counter) + ": reading from " + root + name)

                video = read_video(os.path.join(root, name), PREDICTOR_PATH)
                alignments = read_align(os.path.join(root, '../align/', name.split(".")[0] + ".align"))

                for start, stop, word in alignments:
                    if word == 'sil' or word == 'sp':
                        continue
                   
                    if (len(x) > 0):
                        _, d1, d2, d3 = video[start:stop].shape
                        _, prev_d1, prev_d2, prev_d3 = x[-1].shape
                        if (d1, d2, d3) != (prev_d1, prev_d2, prev_d3):
                            if verbose is True:
                                print("different size, skip")
                            continue
                    
                    x.append(video[start:stop])
                    y.append(word)
                            
                    max_word_len = max(max_word_len, len(word))
                    max_len = max(max_len, stop-start)

                    word_len_list.append(len(word))
                    input_len_list.append(stop-start)
                    
                    counter += 1
                    if counter % num_samples == 0:
                        
                        if not ctc_encoding:
                            y = le.fit_transform(y)
                            y = oh.fit_transform(y.reshape(-1, 1)).todense()

                        for i in range(len(x)):
                            sil_image = get_sil_image()
                            result = np.stack([sil_image for _ in range(max_len)] ,axis = 0)
                            result[:x[i].shape[0], :x[i].shape[1], :x[i].shape[2], :x[i].shape[3]] = x[i]
                            x[i] = result

                            if ctc_encoding:
                                res = np.ones(max_word_len) * -1
                                enc = np.array(text_to_labels(y[i]))
                                res[:enc.shape[0]] = enc
                                y[i] = res

                        if ctc_encoding:
                            y = np.stack(y, axis=0)

                        x = np.stack(x, axis=0)

                        print('saving numpy')
                        x_dir = os.path.join(output_dir, speaker + '_x_' + str(counter / num_samples))
                        y_dir = os.path.join(output_dir, speaker + '_y_' + str(counter / num_samples))
                        wi_dir = os.path.join(output_dir, speaker + '_wi_' + str(counter / num_samples))
                        np.savez_compressed(x_dir, x=x)
                        np.savez_compressed(y_dir, y=y)
                        np.savez_compressed(wi_dir,
                                            word_length=word_len_list, input_length=input_len_list)
                  
                        max_len = 0
                        max_word_len = 0

                        x = list()
                        y = list()

                        word_len_list = []
                        input_len_list = []


    return 1 + counter / num_samples

def read_data_for_speaker(speaker_id, count):
    data_dir = os.path.join("/global/scratch/alex_vlissidis/lipreading_data/", speaker_id + '_np')
    x_dir = os.path.join(data_dir, speaker_id + "_x_" + str(count) + ".npz")
    y_dir = os.path.join(data_dir, speaker_id + "_y_" + str(count) + ".npz")
    wi_dir = os.path.join(data_dir, speaker_id + "_wi_" + str(count) + ".npz")
    try:
        x = np.load(x_dir)['x']
        y = np.load(y_dir)['y']
        word_len = np.load(wi_dir)['word_length']
        input_len = np.load(wi_dir)['input_length']
    except:
        print("make sure save in /path_to_data/data/s*_np/")
        raise
    return x, y, word_len, input_len


if __name__ == "__main__":
    load_data(DATA_PATH, 's1', num_samples = 1000)

