#!/bin/bash

for dimh in 16 32 64 128 #4 8 16 32 64 128
do
for numlayer in 2 4 8 #2 4 8 16
do
for i in 0 1
do
echo $numlayer-$dimh
# CoNSeP
# python hover_net/compute_stats.py --pred_dir output/CoNSeP_train_out/graph/$numlayer-$dimh --true_dir ../data/CoNSeP/Train/Labels --mode type
# python hover_net/compute_stats.py --pred_dir output/CoNSeP_test_out/graph/$numlayer-$dimh --true_dir ../data/CoNSeP/Test/Labels --mode type

python hover_net/compute_stats.py --pred_dir output/Lizard_valid_out/graph/$numlayer-$dimh-acc-macro-$i --true_dir ../data/Lizard_dataset_split/patches/Lizard_Labels_valid --mode type
# python hover_net/compute_stats.py --pred_dir output/Lizard_test_out/graph/$numlayer-$dimh-acc-macro-$i --true_dir ../data/Lizard_dataset_split/patches/Lizard_Labels_test --mode type
done
done
done
echo All done