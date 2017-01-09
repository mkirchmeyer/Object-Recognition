#!/bin/bash
annotations='data/annotations/instances_val2014.json'
main='main.py'


size_word_vec=('50' '100' '200')
mode_word_vec=('area' 'max')
#layer=('fc6' 'fc7' 'fc8' 'prob')
layer=('prob')

output_dir='output'

method='T2I'

for size_word in "${size_word_vec[@]}"
do
	for mode_word in "${mode_word_vec[@]}"
	do
		for lay in "${layer[@]}"
		do		
			glove='data/glove.6B/glove.6B.'$size_word'd.txt'
			train_word_feat='data/cat_feat/'$mode_word'_'$size_word'_train.npy'	
			test_word_feat='data/cat_feat/'$mode_word'_'$size_word'_test.npy'	
			train_img_feat='data/img_feat/'$lay'_train.npy'
			test_img_feat='data/img_feat/'$lay'_test.npy'
            
            title='word_feature_'$size_word'd_'$mode_word'_|_img_feature_'$lay
            
            echo '------------------------------'
            echo 'word_feature_'$size_word'd_'$mode_word'_|_img_feature_'$lay
            echo '------------------------------'
            python $main $method $train_img_feat $train_word_feat $test_img_feat $test_word_feat $glove $annotations -t $title
		done	
	done
done	

