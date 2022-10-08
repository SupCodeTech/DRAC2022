# DRAC2022 DR Segmentation
The project is under construction
## Abstract

People with diabetes are more likely to develop retinal lesions than healthy people. However, retinopathy is the leading cause of blindness. At present, for the diagnosis of diabetic retinopathy, it mainly relies on the experienced clinician to recognize the fine features in color fundus images. This is a time-consuming task. In this paper, to detect diabetic retinopathy in fundus images, we propose a novel semi-supervised UW-OCTA semantic segmentation method for diabetic retinal images, MCS-DRNet. This method, first, uses the MAE algorithm to perform semi-supervised pre-training on the UW-OCTA diabetic retinopathy grade classification dataset to mine the supervised information in the images, thereby alleviating the need for labeled data. Secondly, in order to more fully mine the lesion features of each region in the UW-OCTA image, this paper constructs a Cross-Algorithm Ensemble DR lesion tissue segmentation algorithm by deploying three algorithms with different visual feature processing strategies. The algorithm contains three sub-algorithms, namely pre-trained MAE, ConvNeXt and SegFormer. We validate the effectiveness of MCS-DRNet for identifying DR features in UW-OCTA images on the DRAC2022 semantic segmentation dataset, and the method achieves a good score (0.5544 mean DES). 

<!-- [ABSTRACT] -->

![Figure1](https://user-images.githubusercontent.com/111235455/194624992-cd09e471-d550-4e89-9e3f-3c152463fa0e.jpg)


Stage 1: Pre-task training

Preparation of the dataset catalog

```none
├── Se_sup
│   ├── Data
│   │   ├── pre_training_data
│   │   │   ├── 001.jpg
│   │   │   ├── ...
│   │   ├── pre_train.txt
```

Configure the self-supervised learning environment

```shell
nvcc -V
# Check GCC version
gcc --version
pip install openmim
mim install mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
git clone https://github.com/SupCodeTech/DRAC2022.git
cd DRAC2022//
# Install MMSelfSup from source
pip install -e .
```

```shell
bash python tools/train.py configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_drac2022.py 
```

Extract weight

```shell
bash python tools/model_converters/extract_backbone_weights.py \
  Pre_training_output_dirs/epoch_1600.pth \
  Pre_training_output_dirs/Pre_training_weight_backbone.pth
```

Stage 2: Semantic segmentation

Preparation of a semantically segmented dataset catalog

```none
├── DRAC2022_dataset
│   ├── Segmentation
│   │   ├── Training
│   │   │   ├── A
│   │   │   │   ├── 640
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── 1024
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   │   │   ├── B
│   │   │   │   ├── 640
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── 1024
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   │   ├── Val
│   │   │   ├── A
│   │   │   │   ├── 640
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── 1024
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   │   │   ├── B
│   │   │   │   ├── 640
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── 1024
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 001.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 001.jpg
│   │   │   │   │   │   ├── ...
│   ├── Testset
```

Semantic segmentation environment configuration

```shell
# Check nvcc version
nvcc -V
# Check GCC version
gcc --version
# Install PyTorch
pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# Install MMCV
pip install openmim
mim install mmcv-full==1.6.0
# !rm -rf mmsegmentation
# !git clone https://github.com/open-mmlab/mmsegmentation.git 
cd /content/drive/MyDrive/mmsegmentation-0.24.1
pip install -e .
cd /content/drive/MyDrive/mmsegmentation-0.24.1
```







