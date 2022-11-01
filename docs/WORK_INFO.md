HOVERNET:

What was done:

define a new dataset class for Lizard in dataset.py.

setup the environement:
```
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.6.0 torchvision==0.7.0
```

Extract data :
edit the paths in extract_patches.py, then:
```
python models/hover_net/extract_patches.py 
```

Run training:
```
python models/hover_net/run_train.py --gpu='0'
```


Lizard:

data utils-> data_extraction.py

extract the splited data source into a pkl file with annotations.

```
python sflizard/data_utils/data_extraction.py -ip data/Lizard_dataset/Lizard_Images1 data/Lizard_dataset/Lizard_Images2 -mf data/Lizard_dataset/Lizard_Labels/Labels -of data-540-200.pkl
```

Stardist:

Update model to use pytorch lightning.

training stardist:
```
python sflizard/training.py --data_path=data-540-200.pkl --max_epochs=10 --batch_size=4 --gpus=1
```

Graph:

Graph datamodule with pytorch geometric
https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
