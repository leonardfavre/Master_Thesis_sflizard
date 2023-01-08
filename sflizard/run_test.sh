#!/bin/bash

echo "Starting test" > test_results.txt

for dimh in 256dh # 16dh 32dh 64dh 128dh 256dh 512dh
do
for numlayer in 2lay # 4lay 8lay 16lay
do
for model in graph_sage # graph_gat graph_gin
do
for xtype in 4ll+c # 4ll
do
for distance in 30 35 40 45 50 55 60
do
echo $numlayer-$dimh-$xtype  >> test_results.txt
python sflizard/test_pipeline.py --graph_weights_path models/${model}_${dimh}_${numlayer}_${xtype}_${distance}dist_500epochs_0.0005lr.ckpt >> test_results.txt
done
done
done
done
done
echo All done