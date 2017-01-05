import numpy as np

def get_word_vectors(vocab_file,vectors_file):
    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(vectors_files):
        vectors = {} #empty dictionnary
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = map(float,vals[1:])
    
    # all vector representations are now in vectors
