#!/bin/bash

size_word_vec=('300')
mode_word_vec=('area')
layer=('fc8')

output_dir='output'

method='I2T'
train_img_feat='data/img_feat/'$layer'_train_big.npy'
train_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_train_big.npy'
test_img_feat='data/img_feat/'$layer'_test_big.npy'
test_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_test_big.npy'
#train_img_feat='data/img_feat/'$layer'_train_small.npy'
#train_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_train_small.npy'
#test_img_feat='data/img_feat/'$layer'_test_small.npy'
#test_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_test_small.npy'
query_image_path='query/clock.jpeg'

glove='data/glove.6B/glove.6B.'$size_word_vec'd.txt'
cca='cca/'$mode_word_vec'_'$size_word_vec'_'$layer'.npy'
annotations='data/annotations/instances_val2014.json'
main='main.py'

#python $main $method $train_img_feat $train_word_feat $test_img_feat $test_word_feat $cca $glove $annotations

python $main $method $train_img_feat $train_word_feat $test_img_feat $test_word_feat $cca $glove $annotations -i $query_image_path
