import numpy as np

def load_glove(vectors_file):
    vectors = {} #empty dictionnary
    with open(vectors_file):
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = map(float,vals[1:])
    return vectors, len(vectors[vectors.keys()[0]])
    # all vector representations are now in vectors

class glove():
    def __init__(self,path_to_glove_txt_file):
        self.dict, self.len_vectors = load_glove(path_to_glove_txt_file)

    def vec(self,word):
        if not word in self.dict.keys()):
            raise NameError('%s not in glove dictionnary' % word)
        return self.dict[word]
    
    def vec_expression(self,expression):
        output = np.zeros(self.len_vectors)
        expression = expression.rstrip().split()
        n_words = 0
        for word in expression:
            if word in self.dict.keys():
                n_words += 1
                output = output + self.vec(word)
        if n_words == 0:
            raise NameError('\'%s\' not in glove dictionnary' % expression)
        return output
