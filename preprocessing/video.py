from skvideo.io import vread

def read_video(path, verbose=False):
    if verbose:
        print("loading: " + path)
    video = vread(path)
    return video

if __name__ == "__main__":
    read_video('/Users/alex/machine-lip-reading/train/../data/s1/video/prwq3s.mpg', verbose=True)
