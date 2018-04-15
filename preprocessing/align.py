import numpy as np

def read_align(path_to_align = None):

    with open(path_to_align, 'r') as f:
        lines = f.readlines()	

    align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
    words = []
    words.append('sil')

    for i in range(75):
    	for j in align:
	    	if i > j[0] and i <= j[1]:
	    		words.append(j[2])
    dict_align  = np.array(words)
    
    return dict_align

