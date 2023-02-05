#!/bin/bash

echo "Starting test" > test_results_graph.txt

for checkp in 
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
echo $checkp >> test_results_graph.txt
python sflizard/run_test_pipeline.py --stardist_weights models/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt --graph_weights_path models/cp_acc_graph/final2-graph_sage-512-8-4ll+c-45-0.0005-acc-epoch=33-val_acc=0.7811.ckpt models/cp_acc_graph/final2-graph_sage-512-4-4ll+c-45-0.0005-acc-epoch=70-val_acc=0.7803.ckpt models/cp_acc_graph/final2-graph_sage-256-4-4ll+c-45-0.0005-acc-epoch=121-val_acc=0.7807.ckpt >> test_results_graph.txt #--graph_weights_path models/${model}_${dimh}_${numlayer}_${xtype}_${distance}dist_500epochs_0.0005lr.ckpt >> test_results.txt
done
# done
# done
# done
# done
echo All done