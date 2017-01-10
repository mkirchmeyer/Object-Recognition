#!/bin/bash
#annotations='data/annotations/instances_val2014.json'
#main='main.py'
#
#size_word_vec=('50' '100' '200')
#mode_word_vec=('area' 'max')
##layer=('fc6' 'fc7' 'fc8' 'prob')
#layer=('prob')
#
#output_dir='output'

#method='T2I'

#for size_word in "${size_word_vec[@]}"
#do
#	for mode_word in "${mode_word_vec[@]}"
#	do
#		for lay in "${layer[@]}"
#		do
#			glove='data/glove.6B/glove.6B.'$size_word'd.txt'
#			train_word_feat='data/cat_feat/'$mode_word'_'$size_word'_train.npy'
#			test_word_feat='data/cat_feat/'$mode_word'_'$size_word'_test.npy'
#			train_img_feat='data/img_feat/'$lay'_train.npy'
#			test_img_feat='data/img_feat/'$lay'_test.npy'
#
#            title='word_feature_'$size_word'd_'$mode_word'_|_img_feature_'$lay
#
#            echo '------------------------------'
#            echo 'word_feature_'$size_word'd_'$mode_word'_|_img_feature_'$lay
#            echo '------------------------------'
#            python $main $method $train_img_feat $train_word_feat $test_img_feat $test_word_feat $glove $annotations -t $title
#		done
#	done
#done

#!/bin/bash

size_word_vec=('300' '50' '100' '200')
#size_word_vec=('300')
mode_word_vec=('area' 'max')
layer=('fc6' 'fc7' 'fc8' 'prob')

output_dir='output'

method='I2T'
#train_img_feat='data/img_feat/'$layer'_train_big.npy'
#train_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_train_big.npy'
#test_img_feat='data/img_feat/'$layer'_test_big.npy'
#test_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_test_big.npy'
train_img_feat='data/img_feat/'$layer'_train_small.npy'
train_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_train_small.npy'
test_img_feat='data/img_feat/'$layer'_test_small.npy'
test_word_feat='data/cat_feat/'$mode_word_vec'_'$size_word_vec'_test_small.npy'
#query_image_path='query/pizza.jpeg'

glove='data/glove.6B/glove.6B.'$size_word_vec'd.txt'
annotations='data/annotations/instances_val2014.json'
main='main.py'

python $main $method $train_img_feat $train_word_feat $test_img_feat $test_word_feat $glove $annotations

#python $main $method $train_img_feat $train_word_feat $test_img_feat $test_word_feat $glove $annotations -i $query_image_path
