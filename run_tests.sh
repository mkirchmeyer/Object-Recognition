#!/bin/bash

size_word_vec='100'
mode_word_vec='max'
layer='fc7'

output_dir='output'

train_img_feat='data/img_feat/'$layer'_train_small.npy'
train_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_train_small.npy'
test_img_feat='data/img_feat/'$layer'_test_small.npy'
test_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_test_small.npy'

glove='data/glove.6B/glove.6B.'$size_word_vec'd.txt'
annotations='data/annotations/instances_val2014.json'
main='main.py'

python $main $train_img_feat $train_word_feat $test_img_feat $test_word_feat $glove $annotations 
