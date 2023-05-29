# Diabetic Retinopathy Analysis Challenge 2022 (DRAC2022)


Papers accepted by the [25th International Conference on Medical Image Computing and Computer Assisted Intervention](https://conferences.miccai.org/2022/en/) (MICCAI2022):

`For Task 1 and 3`: 
[Semi-Supervised Semantic Segmentation Methods for UW-OCTA Diabetic Retinopathy Grade Assessment](https://link.springer.com/chapter/10.1007/978-3-031-33658-4_10#citeas)

The methods (MCSDR-Net v1) constructed by my team (Galactic Fleet) was placed 7th (0.5161 mDice) in the DRAC2022 Semantic Segmentation (Task1) challenge.

![image](https://user-images.githubusercontent.com/111235455/222426224-81e1d41b-7aab-48f9-9cac-628bcab4fb9c.png)

After the challenge, we improved the built model  (MCSDR-Net v1)  and proposed a second version  (MCSDR-Net v2) , which has greatly improved the accuracy of the model, reaching 0.5544 mDice.

![image](https://user-images.githubusercontent.com/111235455/222428329-19ad2d5c-c821-496b-8019-237ec3c1ea15.png)


`For Task 2`: [Image Quality Assessment based on Multi-Model Ensemble Class-Imbalance Repair Algorithm for Diabetic Retinopathy UW-OCTA Images](https://link.springer.com/chapter/10.1007/978-3-031-33658-4_11#citeas) 

If our work is helpful to you, please cite the following papers:

```latex

@article{tan2022semi,
   title={Semi-Supervised Semantic Segmentation Methods for UW-OCTA Diabetic Retinopathy Grade Assessment},
   author={Tan, Zhuoyi and Madzin, Hizmawati and Ding, Zeyu},
   journal={arXiv preprint arXiv:2212.13486},
   year={2022}
}

@InProceedings{10.1007/978-3-031-33658-4_10,
author="Tan, Zhuoyi
and Madzin, Hizmawati
and Ding, Zeyu",
editor="Sheng, Bin
and Aubreville, Marc",
title="Semi-supervised Semantic Segmentation Methods for UW-OCTA Diabetic Retinopathy Grade Assessment",
booktitle="Mitosis Domain Generalization and Diabetic Retinopathy Analysis",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="97--117",
abstract="People with diabetes are more likely to develop diabetic retinopathy (DR) than healthy people. However, DR is the leading cause of blindness. At present, the diagnosis of diabetic retinopathy mainly relies on the experienced clinician to recognize the fine features in color fundus images. This is a time-consuming task. Therefore, in this paper, to promote the development of UW-OCTA DR automatic detection, we propose a novel semi-supervised semantic segmentation method for UW-OCTA DR image grade assessment. This method, first, uses the MAE algorithm to perform semi-supervised pre-training on the UW-OCTA DR grade assessment dataset to mine the supervised information in the UW-OCTA images, thereby alleviating the need for labeled data. Secondly, to more fully mine the lesion features of each region in the UW-OCTA image, this paper constructs a cross-algorithm ensemble DR tissue segmentation algorithm by deploying three algorithms with different visual feature processing strategies. The algorithm contains three sub-algorithms, namely pre-trained MAE, ConvNeXt, and SegFormer. Based on the initials of these three sub-algorithms, the algorithm can be named MCS-DRNet. Finally, we use the MCS-DRNet algorithm as an inspector to check and revise the results of the preliminary evaluation of the DR grade evaluation algorithm. The experimental results show that the mean dice similarity coefficient of MCS-DRNet v1 and v2 are 0.5161 and 0.5544, respectively. The quadratic weighted kappa of the DR grading evaluation is 0.7559. Our code is available at https://github.com/SupCodeTech/DRAC2022.",
isbn="978-3-031-33658-4"
}

@InProceedings{10.1007/978-3-031-33658-4_11,
author="Tan, Zhuoyi
and Madzin, Hizmawati
and Ding, Zeyu",
editor="Sheng, Bin
and Aubreville, Marc",
title="Image Quality Assessment Based on Multi-model Ensemble Class-Imbalance Repair Algorithm for Diabetic Retinopathy UW-OCTA Images",
booktitle="Mitosis Domain Generalization and Diabetic Retinopathy Analysis",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="118--126",
abstract="In the diagnosis of diabetic retinopathy (DR), ultrawide optical coherence tomography angiography (UW-OCTA) has received extensive attention because it can non-invasively detect the changes of neovascularization in diabetic retinopathy images. However, in clinical application, UW-OCTA digital images will always suffer a variety of distortions due to a variety of uncontrollable factors, and then affect the diagnostic effect of DR. Therefore, screening images with better imaging quality is very crucial to improve the diagnostic efficiency of DR. In this paper, to promote the development of UW-OCTA DR image quality automatic assessment, we propose a multi-model ensemble class-imbalance repair (MMECIR) algorithm for UW-OCTA DR image quality grading assessment. The models integrated with this algorithm are ConvNeXt, EfficientNet v2, and Xception. The experimental results show that the MMECIR algorithm constructed in this paper can be well applied to UW-OCTA diabetic retinopathy image quality grading assessment (the quadratic weighted kappa of this algorithm is 0.6578). Our code is available at https://github.com/SupCodeTech/DRAC2022.",
isbn="978-3-031-33658-4"
}

```

