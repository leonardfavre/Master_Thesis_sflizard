#!/bin/bash

for dimh in 256 #16 32 64 128 256 512
do
for numlayer in 2
do
for model in graph_sage #graph_sage graph_gat graph_gin
do
for xtype in 4ll+c # 4ll
do
for distance in 30 35 40 45 50 55 60
do
echo $numlayer-$dimh-$distance
python sflizard/training.py --model $model --max_epochs 500 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance
done
done
done
done
done
echo All done