#!/bin/bash

for dimh in 256 512 # 32 64 128 # 256 512
do
for numlayer in 2 4 8
do
for model in graph_custom # graph_gat graph_gin graph_GCN
do
for xtype in 4ll+c # 4ll
do
for distance in 45 # 30 # 45 # 30 45 60
do
echo $numlayer-$dimh-$distance-$xtype-$model
python sflizard/training.py --model $model --max_epochs 200 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance
done
done
done
done
done
echo All done