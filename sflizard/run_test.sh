#!/bin/bash

echo "Starting test" > test_results_stardist_lr2.txt

for checkp in loss_cb/final3-stardist_class-1.0losspower_0.0004lr-crop-cosine-loss-epoch=186-val_loss=3.39.ckpt final3_stardist_crop-cosine_200epochs_1.0losspower_0.0004lr.ckpt loss_cb/final3-stardist_class-1.0losspower_0.0006lr-crop-cosine-loss-epoch=186-val_loss=3.40.ckpt final3_stardist_crop-cosine_200epochs_1.0losspower_0.0006lr.ckpt 
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
echo $checkp >> test_results_stardist_lr2.txt
python sflizard/run_test_pipeline.py --stardist_weights models/${checkp} >> test_results_stardist_lr2.txt #--graph_weights_path models/${model}_${dimh}_${numlayer}_${xtype}_${distance}dist_500epochs_0.0005lr.ckpt >> test_results.txt
done
# done
# done
# done
# done
echo All done