# DRAC2022 (MICCAI2022) DR Image Quality Assessment

`This project is still under construction` 

Before you download the dataset, please request permission from the DRAC2022 organizer ([Official website](https://drac22.grand-challenge.org/)) to access the dataset.

For convenience, we strongly recommend downloading our preprocessed data set (For [training set](https://drive.google.com/file/d/1zA1XwS58fHcAAAQalbQp9QBzGUIMqli5/view?usp=sharing) and [validation set](https://drive.google.com/file/d/1GndQtL1G0a9hEDsVNaiMxTCe4oEaOWx4/view?usp=share_link)). 

Note: due to the large randomness of Mask patch generation, different Mask conditions correspond to different performance. Therefore, using our data set can maximize the avoidance of this problemand. 

When you complete the download process, you need to unzip it. Save the data in the following directory:

```none
├── Data
│   ├── TrainingOAMASK
│   │   ├── High
│   │   │   ├── 885.jpg
│   │   │   ├── ...
│   │   ├── Low
│   │   │   ├── 1.jpg
│   │   │   ├── ...
│   │   ├── Med
│   │   │   ├── 302.jpg
│   │   │   ├── ...
│   ├── ValOA
│   │   ├── High
│   │   │   ├── 3941.jpg
│   │   │   ├── ...
│   │   ├── Low
│   │   │   ├── 289.jpg
│   │   │   ├── ...
│   │   ├── Med
│   │   │   ├── 848.jpg
│   │   │   ├── ...
```

Data preprocessing file `Data_preprocessing.py` , will be released soon.


The environment is configured as follows:
```shell
pip install gdown -q
pip install jax==0.3.13 jaxlib==0.3.10 -q
pip install pyyaml h5py  
```
For ConvNeXt's training, we need to use install tensorflow 2.10.0:
```shell
pip install tensorflow==2.10.0
```
For EfficientNetV2's training, we need to use install tensorflow 2.9.2:
```shell
pip install tensorflow==2.9.2
```
