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
python src/data_utils/data_extraction.py -ip Lizard_dataset/Lizard_Images1 Lizard_dataset/Lizard_Images2 -mf Lizard_dataset/Lizard_Labels/Labels -of data.pkl
```

