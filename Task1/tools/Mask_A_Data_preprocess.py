import os.path as osp
import numpy as np
from PIL import Image
from pandas import Series,DataFrame
import sys
import pickle
import cv2
import os
import tensorflow as tf
import pandas as pd
import shutil
import mmcv
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
# import tensorflow as tf


Resolution = 1024

img_dir = 'DRAC2022_dataset/A._Segmentation/1._Original Images/a._Training Set'
groundtruth = 'DRAC2022_dataset/A._Segmentation/2._Groundtruths/a._Training Set'
img_dir_1 = '1._Intraretinal_Microvascular_Abnormalities'
img_dir_3 = '3._Neovascularization'

palette = [[125, 125, 125], [20, 10, 130], [140, 140, 40], [140, 40, 60]]


import cv2
import os.path
import os
import numpy as np
 
for file in mmcv.scandir(img_dir, suffix='.png'):

  seg_map_1 = np.zeros(1048576).reshape((Resolution, Resolution))
  seg_map_3 = np.zeros(1048576).reshape((Resolution, Resolution))
  
  img_1 = os.path.join(osp.join(groundtruth, img_dir_1), file)
  img_3 = os.path.join(osp.join(groundtruth, img_dir_3), file)

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

  # 1024
  seg_img_1024_ = Image.fromarray(seg_map).convert('P')
  seg_img_1024 = Image.fromarray(seg_map).convert('P')
  seg_img_1024.putpalette(np.array(palette, dtype=np.uint8))

  seg_img_1024_flip_left_right = seg_img_1024_.transpose(Image.FLIP_LEFT_RIGHT)
  seg_img_1024_flip_left_right.putpalette(np.array(palette, dtype=np.uint8))  

  seg_img_1024_flip_top_bottom = seg_img_1024_.transpose(Image.FLIP_TOP_BOTTOM)
  seg_img_1024_flip_top_bottom.putpalette(np.array(palette, dtype=np.uint8))  

  seg_map_rotate_1024_90 = seg_img_1024_.rotate(90, expand=1)
  seg_map_rotate_1024_90.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_1024_180 = seg_img_1024_.rotate(180, expand=1)
  seg_map_rotate_1024_180.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_1024_270 = seg_img_1024_.rotate(270, expand=1)
  seg_map_rotate_1024_270.putpalette(np.array(palette, dtype=np.uint8))

  # 640
  seg_map_640 = seg_map_1 + seg_map_3
  seg_map_640_ = seg_map_1 + seg_map_3

  seg_map_640 = Image.fromarray(seg_map_640).convert('P')
  seg_map_640_ = Image.fromarray(seg_map_640_).convert('P')
  seg_map_640 = seg_map_640.resize((640, 640), Image.ANTIALIAS)
  seg_map_640_ = seg_map_640_.resize((640, 640), Image.ANTIALIAS)
  seg_map_640_.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_640_90 = seg_map_640.rotate(90, expand=1)
  seg_map_rotate_640_90.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_640_180 = seg_map_640.rotate(180, expand=1)
  seg_map_rotate_640_180.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_640_270 = seg_map_640.rotate(270, expand=1)
  seg_map_rotate_640_270.putpalette(np.array(palette, dtype=np.uint8))


  seg_img_1024_path = './DRAC2022_dataset/Segmentation/Training/A/1024/Masks'
  seg_img_1024_path_folder = os.path.exists(seg_img_1024_path)
  if not seg_img_1024_path_folder:
        os.makedirs(seg_img_1024_path)
  seg_img_1024.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", file))
  # Flip horizontal
  seg_img_1024_flip_left_right.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "009" + file))
  # Flip vertical
  seg_img_1024_flip_top_bottom.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "119" + file))
  # rotate 90 640 raw mask saving
  seg_map_rotate_1024_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "909" + file))
  # rotate 180 640 raw mask saving
  seg_map_rotate_1024_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "1809" + file))
  # rotate 270 640 raw mask saving
  seg_map_rotate_1024_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "2709" + file))

  img_1024_path = './DRAC2022_dataset/Segmentation/Training/A/1024/Original_images'
  img_1024_path_folder = os.path.exists(img_1024_path)
  if not img_1024_path_folder:
        os.makedirs(img_1024_path)

  image_1024 = cv2.imread(img_dir + '/' + file, 1)
  image_1024_ = Image.fromarray(image_1024).convert('RGB')
  # raw 1024 image saving
  image_1024_.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", file.replace('.png','.jpg')))
  # Flip horizontal
  image_1024_flip_left_right = image_1024_.transpose(Image.FLIP_LEFT_RIGHT)
  image_1024_flip_left_right.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "009" + file.replace('.png','.jpg')))
  # Flip vertical
  image_1024_flip_top_bottom = image_1024_.transpose(Image.FLIP_TOP_BOTTOM)
  image_1024_flip_top_bottom.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "119" + file.replace('.png','.jpg')))
  # rotate 90 1024 raw image saving
  image_1024_90 = image_1024_.rotate(90, expand=1)
  image_1024_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "909" +file.replace('.png','.jpg')))
  # rotate 180 1024 raw image saving
  image_1024_180 = image_1024_.rotate(180, expand=1)
  image_1024_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "1809" +file.replace('.png','.jpg')))
  # rotate 270 1024 raw image saving
  image_1024_270 = image_1024_.rotate(270, expand=1)
  image_1024_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "2709" +file.replace('.png','.jpg')))
  seg_640_path = "./DRAC2022_dataset/Segmentation/Training/A/640/Masks"
  seg_640_path_folder = os.path.exists(seg_640_path)
  if not seg_640_path_folder:
    os.makedirs(seg_640_path)
  # raw 640 mask saving
  seg_map_640_.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", file))
   # Flip horizontal
  seg_map_640_left_right = seg_map_640_.transpose(Image.FLIP_LEFT_RIGHT)
  seg_map_640_left_right.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "009" + file))
  # Flip vertical
  seg_map_640_flip_top_bottom = seg_map_640_.transpose(Image.FLIP_TOP_BOTTOM)
  seg_map_640_flip_top_bottom.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "119" + file))
  
  # rotate 90 640 raw mask saving
  seg_map_rotate_640_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "909" + file))
  # rotate 180 640 raw mask saving
  seg_map_rotate_640_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "1809" + file))
  # rotate 270 640 raw mask saving
  seg_map_rotate_640_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "2709" + file))
  
  img_640_path = './DRAC2022_dataset/Segmentation/Training/A/640/Original_images'
  img_640_path_folder = os.path.exists(img_640_path)
  if not img_640_path_folder:
        os.makedirs(img_640_path)
  Original_image_640 = cv2.imread(img_dir + '/' + file, 1)
  Original_image_640 = Image.fromarray(Original_image_640).convert('RGB')
  Original_image_640 = Original_image_640.resize((640, 640), Image.ANTIALIAS)
  # raw 640 image saving
  Original_image_640.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", file.replace('.png','.jpg')))

  # Flip horizontal
  image_640_flip_left_right = Original_image_640.transpose(Image.FLIP_LEFT_RIGHT)
  image_640_flip_left_right.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "009" + file.replace('.png','.jpg')))
  # Flip vertical
  image_640_flip_top_bottom = Original_image_640.transpose(Image.FLIP_TOP_BOTTOM)
  image_640_flip_top_bottom.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "119" + file.replace('.png','.jpg')))

  # rotate 90 640 raw image saving
  image_640_90 = Original_image_640.rotate(90, expand=1)
  image_640_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "909" +file.replace('.png','.jpg')))
  # rotate 180 640 raw image saving
  image_640_180 = Original_image_640.rotate(180, expand=1)
  image_640_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "1809" +file.replace('.png','.jpg')))
  # rotate 270 640 raw image saving
  image_640_270 = Original_image_640.rotate(270, expand=1)
  image_640_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "2709" +file.replace('.png','.jpg')))

