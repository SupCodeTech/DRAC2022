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

img_dir = './DRAC2022_dataset/Test_Data/Original_Images'

for file in mmcv.scandir(img_dir, suffix='.png'):
  
  img = os.path.join(osp.join(img_dir, file))

  image_tensor = cv2.imread(img, 2)
  image_tensor = Image.fromarray(image_tensor).convert('RGB')

  Horizontal_flip = './DRAC2022_dataset/Test_Data/Horizontal_flip'
  Horizontal_flip_folder = os.path.exists(Horizontal_flip)
  if not Horizontal_flip_folder:
    os.makedirs(Horizontal_flip)

  Vertical_flip = './DRAC2022_dataset/Test_Data/Vertical_flip'
  Vertical_flip_folder = os.path.exists(Vertical_flip)
  if not Vertical_flip_folder:
    os.makedirs(Vertical_flip)

  R90 = './DRAC2022_dataset/Test_Data/R90'
  R90_folder = os.path.exists(R90)
  if not R90_folder:
    os.makedirs(R90)

  R180 = './DRAC2022_dataset/Test_Data/R180'
  R180_folder = os.path.exists(R180)
  if not R180_folder:
    os.makedirs(R180)

  R270 = './DRAC2022_dataset/Test_Data/R270'
  R270_folder = os.path.exists(R270)
  if not R270_folder:
    os.makedirs(R270)

  image_tensor_90 = image_tensor.rotate(90, expand=1)
  image_tensor_90.save(osp.join(R90, file))
  image_tensor_180 = image_tensor.rotate(180, expand=1)
  image_tensor_180.save(osp.join(R180, file))
  image_tensor_270 = image_tensor.rotate(270, expand=1)
  image_tensor_270.save(osp.join(R270, file))

  # Flip horizontal
  image_tensor_left_right = image_tensor.transpose(Image.FLIP_LEFT_RIGHT)
  image_tensor_left_right.save(osp.join(Horizontal_flip, file))
  # Flip Vertical
  image_tensor_flip_top_bottom = image_tensor.transpose(Image.FLIP_TOP_BOTTOM)
  image_tensor_flip_top_bottom.save(osp.join(Vertical_flip, file))
