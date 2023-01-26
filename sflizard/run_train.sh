#!/bin/bash

for dimh in 16 32 64 128 256 512
do
for numlayer in 2 4 8
do
for model in graph_GCN graph_sage graph_gat graph_gin # graph_GCN graph_custom
do
for xtype in ll+c # 4ll
do
for distance in 45 # 30 # 45 # 30 45 60
do
echo $model-$dimh-$numlayer-$xtype-$distance
python sflizard/training.py --model $model --max_epochs 200 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance
done
done
done
done
done
echo All done

python sflizard/training.py --model graph_gat --max_epochs 200 --gpus 1 --num_layers 8 --dimh 512 --x_type "ll" --distance 45 --batch_size 32