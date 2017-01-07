import numpy as np

def load_glove(vectors_file):
    vectors = {} #empty dictionnary
    with open(vectors_file):
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = map(float,vals[1:])
    return vectors
    # all vector representations are now in vectors

class glove():
    def __init__(self,path_to_glove_txt_file):
        self.dict = load_glove(path_to_glove_txt_file)

    def vectorize(self,word):
        if not word in self.dict.keys()):
            raise NameError('%s not in glove dictionnary' % word)
        return self.dict[word]
