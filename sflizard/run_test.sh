#!/bin/bash

echo "Starting test" > test_results_stardist_aug.txt

for checkp in final_stardist_shuffle_noaug_200epochs_0.0losspower_0.0005lr.ckpt stardist_shuffle_rotate_200epochs_0.0losspower_0.0005lr.ckpt final_stardist_shuffle_1000epochs_0.0losspower_0.0005lr.ckpt final_stardist_shuffle_crop_1000epochs_0.0losspower_0.0005lr.ckpt loss_cb/stardist_class-shuffle-noaug-loss-epoch=161-val_loss=1.92.ckpt loss_cb/stardist_class-shuffle-rotate-loss-epoch=151-val_loss=2.05.ckpt loss_cb/stardist_class-shuffle-loss-epoch=141-val_loss=2.04.ckpt loss_cb/stardist_class-shuffle-crop-loss-epoch=165-val_loss=2.08.ckpt 
do


# for dimh in 256dh # 16dh 32dh 64dh 128dh 256dh 512dh
# do
# for numlayer in 2lay # 4lay 8lay 16lay
# do
# for model in graph_sage # graph_gat graph_gin
# do
# for xtype in 4ll+c # 4ll
# do
# for distance in 30 35 40 45 50 55 60
# do
#echo $numlayer-$dimh-$xtype  >> test_results.txt
echo $checkp >> test_results_stardist_aug.txt
python sflizard/test_pipeline.py --stardist_weights models/${checkp} >> test_results_stardist_aug.txt #--graph_weights_path models/${model}_${dimh}_${numlayer}_${xtype}_${distance}dist_500epochs_0.0005lr.ckpt >> test_results.txt
done
# done
# done
# done
# done
echo All done