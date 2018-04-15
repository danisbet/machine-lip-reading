from skvideo.io import vread

def read_video(path, verbose=False):
    if verbose:
        print("loading: " + path)
    video = vread(path)
    return video

if __name__ == "__main__":
    read_video('../data/s1/video/srbizp.mpg', verbose=True)
