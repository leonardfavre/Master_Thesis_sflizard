#!/bin/bash

# for dimh in 1024 # 256 512 1024 2048 # 256 512 1024 # 32 64 128 256
# do
# for numlayer in 4 # 16 8  # 2 4 8
# do
# for model in graph_gat # graph_sage graph_gin graph_GCN # graph_custom graph_gat
# do
# for xtype in 4ll+c # 4ll+c
# do
# for distance in 45 # 30 # 45 # 30 45 60
# do
# for dropout in 0.4 0.5 # 0.1 0.2 0.3 0.4 0.5
# do
# for head in 8 # 2 4 8 # 1 2 4 8
# do
# echo $model-$dimh-$numlayer-$xtype-$distance-$head-$dropout
# python sflizard/training.py --model $model --max_epochs 200 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance --heads $head --batch_size 32 --dropout $dropout
# done
# done
# done
# done
# done
# done
# done

# echo All done


# version for custom graph
for dimh in 1024 # 256 512 1024 2048 # 256 512 1024 # 32 64 128 256
do
for numlayer in 4 # 16 8  # 2 4 8
do
for model in graph_custom # graph_sage graph_gin graph_GCN # graph_custom graph_gat
do
for xtype in 4ll # 4ll+c
do
for distance in 45 # 30 # 45 # 30 45 60
do
for cil in 0 # 1 # 2 3
do
for cih in 0 # 540 1024 2048 
do
for col in 3 # 2 3
do
for coh in 16 # 32
do
for cwc in True
do
for dropout in 0.1 0.2 0.3 0.4 0.5
do
if [[ ( $cil -eq 0  &&  $cih -eq 0 ) || ( $col -eq 0 && $coh -eq 0 )]]
then
# test that at les 1 of the 2 is not 0
# if [[ ( $cil -ne 0 && $cil -ne 1  &&  $cih -ne 0 ) || ( $col -ne 0 && $col -ne 1 && $coh -ne 0 )]]
if [[ ( $cil -ne 0  &&  $cih -ne 0 ) || ( $col -ne 0 && $coh -ne 0 )]]
then
echo $model-$dimh-$numlayer-$xtype-$distance-$cil-$cih-$col-$coh
python sflizard/training.py --custom_input_layer $cil --custom_input_hidden $cih --custom_output_layer $col --custom_output_hidden $coh --model $model --max_epochs 200 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance --batch_size 64 --custom_wide_connections $cwc --dropout $dropout
# elif [[ ( $cil -eq 1  &&  $cih -eq 0 ) || ( $col -eq 1 && $coh -eq 0 )]]
# then
# echo $model-$dimh-$numlayer-$xtype-$distance-$cil-$cih-$col-$coh
# python sflizard/training.py --custom_input_layer $cil --custom_input_hidden $cih --custom_output_layer $col --custom_output_hidden $coh --model $model --max_epochs 200 --gpus 1 --num_layers $numlayer --dimh $dimh --x_type $xtype --distance $distance --batch_size 64 --custom_wide_connections $cwc
# fi
fi
fi
done
done
done
done
done
done
done
done
done
done
done
echo All done