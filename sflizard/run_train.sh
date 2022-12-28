#!/bin/bash

for dimh in 16 32 64 128 256 512
do
for numlayer in 2 4 8 16
do
for model in graph_gat #graph_sage graph_gat graph_gin
do
echo $numlayer-$dimh
python sflizard/training.py --model $model --max_epochs 500 --gpus 1 --num_layers $numlayer --dimh $dimh

done
done
done
echo All done