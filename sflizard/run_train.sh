#!/bin/bash

for dimh in 256 512 1024 2048 # 32 64 128 256
do
for numlayer in 8 16 32 64 # 2 4 8
do
for model in graph_sage # graph_sage graph_gin graph_GCN # graph_custom graph_gat
do
for xtype in 4ll+c
do
for distance in 45 # 30 # 45 # 30 45 60
do
for head in 1 # 1 2 4 8
do
echo $model-$dimh-$numlayer-$xtype-$distance-$head
python sflizard/training.py --model $model --max_epochs 200 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance --heads $head
done
done
done
done
done
done

echo All done
