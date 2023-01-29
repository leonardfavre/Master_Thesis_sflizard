#!/bin/bash

for dimh in 1024 # 256 512 1024 2048 # 256 512 1024 # 32 64 128 256
do
for numlayer in 4 # 16 8  # 2 4 8
do
for model in graph_gat # graph_sage graph_gin graph_GCN # graph_custom graph_gat
do
for xtype in 4ll+c # 4ll+c
do
for distance in 60 50 40 30 # 30 # 45 # 30 45 60
do
for head in 8 # 2 4 8 # 1 2 4 8
do
echo $model-$dimh-$numlayer-$xtype-$distance-$head
python sflizard/training.py --model $model --max_epochs 200 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance --heads $head --batch_size 32
done
done
done
done
done
done

echo All done
