import os.path as osp
import numpy as np
from PIL import Image
from pandas import Series,DataFrame
import sys
import pickle
from pandas import Series,DataFrame
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import pandas as pd
import shutil
import os.path

Resolution = 1024

number_of_training = 80 # set the number of images for training, and the rest for testing.


img_dir = './DRAC2022_dataset/A. Segmentation/1. Original Images/a. Training Set'
groundtruth = './DRAC2022_dataset/A. Segmentation/2. Groundtruths/a. Training Set'
img_dir_1 = '1. Intraretinal Microvascular Abnormalities'
img_dir_3 = '3. Neovascularization'

CLASSES = ('Background', 'Intraretinal_microvascular_abnormals', 'Neovascularization')
PALETTE = [[20, 20, 20], [30, 30, 30], [40, 40, 40]]

spilt = 0


 
 
def img_resize(img):
    height, width = img.shape[0], img.shape[1]
    width_new = 640
    height_new = 640
    # Determine the ratio of length to width of the picture
    if width / height >= width_new / height_new:
        img_new = cv2.resize(img, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(img, (int(width * height_new / height), height_new))
    return img_new
 
for file in mmcv.scandir(img_dir, suffix='.png'):

  seg_map_1 = np.zeros(1048576).reshape((Resolution, Resolution))
  seg_map_3 = np.zeros(1048576).reshape((Resolution, Resolution))
  
  img_1 = os.path.join(osp.join(groundtruth, img_dir_1) + file)
  img_3 = os.path.join(osp.join(groundtruth, img_dir_3) + file)

  if os.path.exists(img_1): 
    image_tensor_1 = cv2.imread(img_1, 2)
    for i in range(Resolution):
      for j in range(Resolution):
        if image_tensor_1[i][j] > 0:
          seg_map_1[i][j] = 1 

  if os.path.exists(img_3): 
    image_tensor_3 = cv2.imread(img_3, 2)
    for i in range(Resolution):
      for j in range(Resolution):
        if image_tensor_3[i][j] > 0:
          seg_map_3[i][j] = 2 

  seg_map = seg_map_1 + seg_map_3

  seg_img = Image.fromarray(seg_map).convert('P')
  seg_img.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_640 = img_resize(seg_map)
  seg_img_640 = Image.fromarray(seg_map_640).convert('P')
  seg_img_640.putpalette(np.array(palette, dtype=np.uint8))

  spilt = spilt + 1

  if spilt < number_of_training:
    seg_img.save("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks")
    seg_img_640.save("./DRAC2022_dataset/Segmentation/Training/A/640/Masks")

    image_640 = cv2.imread(img_dir + file, 1)
    image_640 = Image.fromarray(image_640).convert('RGB')
    seg_img.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", file.replace('.png','.jpg')))
    shutil.copyfile(img_dir + file, osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", file.replace('.png','.jpg')))

  else:
    seg_img.save("./DRAC2022_dataset/Segmentation/Val/A/1024/Masks")
    seg_img_640.save("./DRAC2022_dataset/Segmentation/Val/A/640/Masks")

    image_640 = cv2.imread(img_dir + file, 1)
    image_640 = Image.fromarray(image_640).convert('RGB')
    seg_img.save(osp.join("./DRAC2022_dataset/Segmentation/Val/A/640/Original_images", file.replace('.png','.jpg')))
    shutil.copyfile(img_dir + file, osp.join("./DRAC2022_dataset/Segmentation/Val/A/1024/Original_images", file.replace('.png','.jpg')))
