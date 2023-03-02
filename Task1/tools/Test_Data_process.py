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
import os.path
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
# import tensorflow as tf


img_dir = './DRAC2022_dataset/Test_Data/1024/Original_Images'

for file in mmcv.scandir(img_dir, suffix='.png'):
  
  img = os.path.join(osp.join(img_dir, file))

  image_tensor = cv2.imread(img, 2)
  image_tensor = Image.fromarray(image_tensor).convert('RGB')

  Original_images = './DRAC2022_dataset/Test_Data/1536/Original_Images'
  Original_images_folder = os.path.exists(Original_images)
  if not Original_images_folder:
    os.makedirs(Original_images)

  Horizontal_flip = './DRAC2022_dataset/Test_Data/1536/Horizontal_flip'
  Horizontal_flip_folder = os.path.exists(Horizontal_flip)
  if not Horizontal_flip_folder:
    os.makedirs(Horizontal_flip)

  Horizontal_flip_ = './DRAC2022_dataset/Test_Data/1024/Horizontal_flip'
  Horizontal_flip_folder_ = os.path.exists(Horizontal_flip_)
  if not Horizontal_flip_folder_:
    os.makedirs(Horizontal_flip_)


  Vertical_flip = './DRAC2022_dataset/Test_Data/1536/Vertical_flip'
  Vertical_flip_folder = os.path.exists(Vertical_flip)
  if not Vertical_flip_folder:
    os.makedirs(Vertical_flip)

  Vertical_flip_ = './DRAC2022_dataset/Test_Data/1024/Vertical_flip'
  Vertical_flip_folder_ = os.path.exists(Vertical_flip_)
  if not Vertical_flip_folder_:
    os.makedirs(Vertical_flip_)

  R90 = './DRAC2022_dataset/Test_Data/1536/R90'
  R90_folder = os.path.exists(R90)
  if not R90_folder:
    os.makedirs(R90)

  R90_ = './DRAC2022_dataset/Test_Data/1024/R90'
  R90_folder_ = os.path.exists(R90_)
  if not R90_folder_:
    os.makedirs(R90_)

  R180 = './DRAC2022_dataset/Test_Data/1536/R180'
  R180_folder = os.path.exists(R180)
  if not R180_folder:
    os.makedirs(R180)

  R180_ = './DRAC2022_dataset/Test_Data/1024/R180'
  R180_folder_ = os.path.exists(R180_)
  if not R180_folder_:
    os.makedirs(R180_)

  R270 = './DRAC2022_dataset/Test_Data/1536/R270'
  R270_folder = os.path.exists(R270)
  if not R270_folder:
    os.makedirs(R270)

  R270_ = './DRAC2022_dataset/Test_Data/1024/R270'
  R270_folder_ = os.path.exists(R270_)
  if not R270_folder_:
    os.makedirs(R270_)

  image_tensor_0 = image_tensor.resize((1536, 1536), Image.ANTIALIAS)
  image_tensor_0.save(osp.join(Original_images, file))

  image_tensor_90 = image_tensor.rotate(90, expand=1)
  image_tensor_90 = image_tensor_90.resize((1536, 1536), Image.ANTIALIAS)
  image_tensor_90.save(osp.join(R90, file))

  image_tensor_90_ = image_tensor.rotate(90, expand=1)
  image_tensor_90_.save(osp.join(R90_, file))

  image_tensor_180 = image_tensor.rotate(180, expand=1)
  image_tensor_180 = image_tensor_180.resize((1536, 1536), Image.ANTIALIAS)
  image_tensor_180.save(osp.join(R180, file))

  image_tensor_180_ = image_tensor.rotate(180, expand=1)
  image_tensor_180_.save(osp.join(R180_, file))

  image_tensor_270 = image_tensor.rotate(270, expand=1)
  image_tensor_270 = image_tensor_270.resize((1536, 1536), Image.ANTIALIAS)
  image_tensor_270.save(osp.join(R270, file))

  image_tensor_270_ = image_tensor.rotate(270, expand=1)
  image_tensor_270_.save(osp.join(R270_, file))

  # Flip horizontal
  image_tensor_left_right = image_tensor.transpose(Image.FLIP_LEFT_RIGHT)
  image_tensor_left_right = image_tensor_left_right.resize((1536, 1536), Image.ANTIALIAS)
  image_tensor_left_right.save(osp.join(Horizontal_flip, file))

  image_tensor_left_right_ = image_tensor.transpose(Image.FLIP_LEFT_RIGHT)
  image_tensor_left_right_.save(osp.join(Horizontal_flip_, file))

  # Flip Vertical
  image_tensor_flip_top_bottom = image_tensor.transpose(Image.FLIP_TOP_BOTTOM)
  image_tensor_flip_top_bottom = image_tensor_flip_top_bottom.resize((1536, 1536), Image.ANTIALIAS)
  image_tensor_flip_top_bottom.save(osp.join(Vertical_flip, file))

  image_tensor_flip_top_bottom_ = image_tensor.transpose(Image.FLIP_TOP_BOTTOM)
  image_tensor_flip_top_bottom_.save(osp.join(Vertical_flip_, file))
