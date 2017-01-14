#!/bin/bash
main='train_cca.py'

size_word_vec=('50' '100' '200' '300')
mode_word_vec=('area' 'max')
layer=('fc6' 'fc7' 'fc8' 'prob')
#layer=('fc6')

output_dir='cca/'

method='T2I'

for size_word in "${size_word_vec[@]}"
do
	for mode_word in "${mode_word_vec[@]}"
	do
		for lay in "${layer[@]}"
		do
			train_word_feat='data/cat_feat/'$mode_word'_'$size_word'_train.npy'
			train_img_feat='data/img_feat/'$lay'_train.npy'

           output_path=$output_dir$mode_word'_'$size_word'_'$lay'.npy'

           echo '------------------------------'
           echo 'word_feature_'$size_word'd_'$mode_word'_|_img_feature_'$lay
           echo '------------------------------'
           python $main $train_img_feat $train_word_feat $output_path
        done
	done
done
