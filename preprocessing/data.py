import os
from align import read_align
from video import read_video

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_PATH + '/../data'
PREDICTOR_PATH = CURRENT_PATH + '/shape_predictor_68_face_landmarks.dat'

def load_data(verbose=False, framebyframe=False):
    for root, dirs, files in os.walk(DATA_PATH):
        for name in files:
            if '.mpg' in name:
                if verbose is True:
                    print("reading: " + root)

                video = read_video(os.path.join(root, name), PREDICTOR_PATH)
                words = read_align(os.path.join(root, '../align/', name.split(".")[0] + ".align"))
               
                if verbose is True:
                    print("video shape:", video.shape)
                    print("alignments shape:", words.shape)

                if framebyframe:
                    for frame, word in zip(video, words):
                        yield frame, word
                else:
                    yield video, words

if __name__ == "__main__":
    for img, word in load_data(verbose=True, framebyframe=True):
        #print(img, word)
