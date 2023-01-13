# DRAC2022 (MICCAI2022) DR Image Quality Assessment

`This project is still under construction` 

Before you download the dataset, please request permission from the DRAC2022 organizer ([Official website](https://drac22.grand-challenge.org/)) to access the dataset.

For convenience, we strongly recommend downloading our preprocessed data set (For [training set](https://drive.google.com/file/d/1zA1XwS58fHcAAAQalbQp9QBzGUIMqli5/view?usp=sharing) and [test set](https://drive.google.com/file/d/1GndQtL1G0a9hEDsVNaiMxTCe4oEaOWx4/view?usp=share_link)), and unzip it. Save the data in the following directory:

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

Note: Due to the large randomness of Mask patch generation, different Mask conditions correspond to different performance. Therefore, using our data set can maximize the avoidance of this problem.


