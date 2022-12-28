#!/bin/bash

for dimh in 4 8 16 32 64 128
do
for numlayer in 2 4 8 16
do
echo $numlayer-$dimh
python hover_net/compute_stats.py --pred_dir hover_net/Lizard_test_out/graph/$numlayer-$dimh --true_dir ../data/Lizard_dataset_test_split/Lizard_Labels_test/Labels --mode type

done
done
echo All done