# DRAC2022 DR Segmentation

`This project is still under construction`

## Abstract

People with diabetes are more likely to develop retinal lesions than healthy people. However, retinopathy is the leading cause of blindness. At present, for the diagnosis of diabetic retinopathy, it mainly relies on the experienced clinician to recognize the fine features in color fundus images. This is a time-consuming task. In this paper, to detect diabetic retinopathy in fundus images, we propose a novel semi-supervised UW-OCTA semantic segmentation method for diabetic retinal images, MCS-DRNet. This method, first, uses the MAE algorithm to perform semi-supervised pre-training on the UW-OCTA diabetic retinopathy grade classification dataset to mine the supervised information in the images, thereby alleviating the need for labeled data. Secondly, in order to more fully mine the lesion features of each region in the UW-OCTA image, this paper constructs a Cross-Algorithm Ensemble DR lesion tissue segmentation algorithm by deploying three algorithms with different visual feature processing strategies. The algorithm contains three sub-algorithms, namely pre-trained MAE, ConvNeXt and SegFormer. We validate the effectiveness of MCS-DRNet for identifying DR features in UW-OCTA images on the DRAC2022 semantic segmentation dataset, and the method achieves a good score (0.5544 mean DES). 

<!-- [ABSTRACT] -->

The code is based on [MMSegmentaion v0.24.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1).

# Stage I: Pre-task training

Download the DRAC Task 3 dataset and unzip it. Save the data in the following directory:

```none
├── Se_sup
│   ├── C._Diabetic_Retinopathy_Grading
│   │   ├── 1._Original_Images
│   │   │   ├── a._Training_Set
│   │   │   │   ├── 001.png
│   │   │   │   ├── ...
│   │   ├── 2._Groundtruths
│   │   │   ├── a._DRAC2022_Diabetic_Retinopathy_Grading_Training_Labels.csv
```

After preprocessing the data, the data will be saved in the following directory. (This data preprocessing file will be published soon)

```none
├── Se_sup
│   ├── Data
│   │   ├── Original_Images
│   │   │   ├── Training_Set
│   │   │   │   ├── 001.png
│   │   │   │   ├── ...
│   │   ├── Pretrained_files.txt
```
The pre-training environment is configured as follows:
```shell
pip install openmim
mim install mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
cd ./Se_sup/
pip install -e .
```
Next, we run the following statement to begin our pre-training：

```shell
python ./configs/selfsup/mae/mae.py
```

After the above statement runs, the pre-trained MAE model will be saved in the following directory:

```none
├── Se_sup
│   ├── work_dirs
│   │   ├── mae
│   │   │   ├── epoch_16000.pth
```

Then, please run the following shell statement in the `Se_sup` directory to obtain the backbone of pre-trained MAE-ViT.

```shell
python tools/model_converters/extract_backbone_weights_mae_vit.py
```

After running, the model is saved in the following directory：

```none
├── Se_sup
│   ├── work_dirs
│   │   ├── mae
│   │   │   ├── pretrain_backbone_16k.pth
```
## Another way to get the pretrained mae-vit backbone.
To fine-tune with **multi-node distributed training**, run the following on 2 nodes with 2 GPUs each:
```
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 2 \
    --batch_size 96 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 1600 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
- Install submitit (`pip install submitit`) first.
- Here the effective batch size is 96 (`batch_size` per gpu) * 2 (`nodes`) * 2 (gpus per node) = 384.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
The detailed implementation process can refer to [this document](https://github.com/facebookresearch/mae).

# Stage II: Semantic segmentation
## Semantic segmentation data preprocessing

For Mask A and B, we provide the data preprocessing file `Mask_A_Data_preprocess.py` and `Mask_B_Data_preprocess.py` to preprocess the original data from DRAC2022 Task 1. Before you do that, you need to downlaod the [Task 1](https://drive.google.com/file/d/1tCsqvsowjEsnyomcBTb600xLI26Y1Pxt/view?usp=sharing) data. Then, unzip and put into the folloing directory:

```none
├── DRAC2022_dataset
│   ├── A._Segmentation
│   │   ├── 1._Original Images
│   │   │   ├── a._Training Set
│   │   │   │   ├── 065.png
│   │   │   │   ├── ...
│   │   ├── 2._Groundtruths
│   │   │   ├── a._Training Set
│   │   │   │   ├── 1._Intraretinal Microvascular Abnormalities
│   │   │   │   │   ├── 082.png
│   │   │   │   │   ├── ...
│   │   │   │   ├── 2._Nonperfusion Areas
│   │   │   │   │   ├── 065.png
│   │   │   │   │   ├── ...
│   │   │   │   ├── 3._Neovascularization
│   │   │   │   │   ├── 082.png
│   │   │   │   │   ├── ...
```

For example, for Mask A usage:

```shell
python tools/Mask_A_Data_preprocess.py
```

## The data directory for the semantic segmentation task

The data for this directory can be generated by the `Mask_A_Data_preprocess.py` and `Mask_B_Data_preprocess.py` file.

```none
├── DRAC2022_dataset
│   ├── A. Segmentation
│   │   ├── ...
│   ├── Segmentation
│   │   ├── Training
│   │   │   ├── A
│   │   │   │   ├── 640
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 082.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 082.jpg
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── 1024
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 082.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 082.jpg
│   │   │   │   │   │   ├── ...
│   │   │   ├── B
│   │   │   │   ├── 640
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 065.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 065.jpg
│   │   │   │   │   │   ├── ...
│   │   │   │   ├── 1024
│   │   │   │   │   ├── Masks
│   │   │   │   │   │   ├── 065.png
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── Original_images
│   │   │   │   │   │   ├── 065.jpg
│   │   │   │   │   │   ├── ...
```
Due to the limited data available for this contest, we did not set the validation set.

## Semantic segmentation environment configuration

```shell
nvcc -V
gcc --version
pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
mim install mmcv-full==1.6.0
cd ./DRAC2022
pip install -e .
```

## Running the example script of different sub-algorithm in training Mask A:
Usage:
```shell
python tools/train.py configs/MCS_DRNet/(sub-algorithm)_Task_1_Mask_A_(640x640/1024x1024).py
```

The sub-algorithm contains three values: `M`, `C`, and `S`. The resolution `1024x1024` is unique to the `S` algorithm.
For example, we want to run the subalgorithm `M`.

Usage:
```shell
python tools/train.py configs/MCS_DRNet/M_Task_1_Mask_A_640x640.py
```
## Running the example script of sub-algorithm C in training Mask B:

Usage:
```shell
python tools/train.py configs/MCS_DRNet/C_Task_1_Mask_B_640x640.py
```
## Testing phase

For the tests of Mask A and B, we provide the test file `MCS_DRNet_Task_1_Mask_A_1536x1536.py` and `MCS_DRNet_Task_1_Mask_B_1536x1536.py`(The document will soon be published).

For mask A, usage:
```shell
python tools/MCS_DRNet_Task_1_Mask_A_1536x1536.py --load-from-checkpoint-M ${file_dir} --load-from-checkpoint-C ${file_dir} --load-from-checkpoint-S ${file_dir} --data-dir ${file_dir} --output-data-dir ${file_dir} 
```
For mask B, usage:
```shell
python tools/MCS_DRNet_Task_1_Mask_B_1536x1536.py --load-from-checkpoint-M ${file_dir} --load-from-checkpoint-C ${file_dir} --load-from-checkpoint-S ${file_dir} --data-dir ${file_dir} --output-data-dir ${file_dir} 
```
`--load-from-checkpoint-M`: path to store checkpoints of the pretrained model MAE \
`--load-from-checkpoint-C`: path to store the checkpoints of the pretrained model ConvNeXt \
`--load-from-checkpoint-S`: path to store checkpoints of the pretrained model SegFormer \
`--data-dir`: path to test image \
`--output-data-dir`: output path of test segmentation results

## Contact
If you have any question, please feel free to contact me via tan.joey@student.upm.edu.my

## Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [MAE](https://github.com/facebookresearch/mae),      [Segformer](https://github.com/NVlabs/SegFormer) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.

