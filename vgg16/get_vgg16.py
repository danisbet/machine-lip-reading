import numpy as np
# import tensorflow as tf
import download
import os

data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"
data_dir = "vgg16/"
path_graph_def = "vgg16.tfmodel"


print("Downloading Pre-trained VGG16 Model ...")
download.maybe_download_and_extract(url=data_url, download_dir=data_dir)
