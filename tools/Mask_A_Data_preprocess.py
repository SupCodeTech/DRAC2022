import os.path as osp
import numpy as np
from PIL import Image
from pandas import Series,DataFrame
import sys
import pickle
import pandas as pd
import cv2
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import shutil
import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import sys
import pickle
import pandas as pd
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
# import tensorflow as tf
import pandas as pd

Resolution = 1024

img_dir = 'DRAC2022_dataset/A._Segmentation/1._Original Images/a._Training Set'
groundtruth = 'DRAC2022_dataset/A._Segmentation/2._Groundtruths/a._Training Set'
img_dir_1 = '1._Intraretinal Microvascular Abnormalities'
img_dir_3 = '3._Neovascularization'

palette = [[20, 20, 20], [30, 30, 30], [40, 40, 40]]

spilt = 0

import cv2
import os.path
import os
import numpy as np
 
 
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
  
  # 1024
  seg_img_1024_ = Image.fromarray(seg_map).convert('P')
  seg_img_1024 = Image.fromarray(seg_map).convert('P')
  seg_img_1024.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_1024_90 = seg_img_1024_.rotate(90, expand=1)
  seg_map_rotate_1024_90.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_1024_180 = seg_img_1024_.rotate(180, expand=1)
  seg_map_rotate_1024_180.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_1024_270 = seg_img_1024_.rotate(270, expand=1)
  seg_map_rotate_1024_270.putpalette(np.array(palette, dtype=np.uint8))

  # 640 
  seg_map_640 = img_resize(seg_map)
  seg_map_640_ = img_resize(seg_map)
  seg_map_640 = Image.fromarray(seg_map_640).convert('P')
  seg_map_640_ = Image.fromarray(seg_map_640_).convert('P')
  seg_map_640_.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_640_90 = seg_map_640.rotate(90, expand=1)
  seg_map_rotate_640_90.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_640_180 = seg_map_640.rotate(180, expand=1)
  seg_map_rotate_640_180.putpalette(np.array(palette, dtype=np.uint8))

  seg_map_rotate_640_270 = seg_map_640.rotate(270, expand=1)
  seg_map_rotate_640_270.putpalette(np.array(palette, dtype=np.uint8))

  spilt = spilt + 1

  os.makedirs("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks")
  seg_img_1024.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", file))
  # rotate 90 640 raw mask saving
  seg_map_rotate_1024_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "90_" + file))
  # rotate 180 640 raw mask saving
  seg_map_rotate_1024_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "180_" + file))
  # rotate 270 640 raw mask saving
  seg_map_rotate_1024_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Masks", "270_" + file))
    
  os.makedirs("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images")
  image_1024 = cv2.imread(img_dir + '/' + file, 1)
  image_1024_ = Image.fromarray(image_1024).convert('RGB')
  # raw 1024 image saving
  image_1024_.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", file.replace('.png','.jpg')))
  # rotate 90 1024 raw image saving
  image_1024_90 = image_1024_.rotate(90, expand=1)
  image_1024_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "90_" +file.replace('.png','.jpg')))
  # rotate 180 1024 raw image saving
  image_1024_180 = image_1024_.rotate(180, expand=1)
  image_1024_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "180_" +file.replace('.png','.jpg')))
  # rotate 270 1024 raw image saving
  image_1024_270 = image_1024_.rotate(270, expand=1)
  image_1024_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/1024/Original_images", "270_" +file.replace('.png','.jpg')))

  os.makedirs("./DRAC2022_dataset/Segmentation/Training/A/640/Masks")
  # raw 640 mask saving
  seg_map_640_.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", file))
  # rotate 90 640 raw mask saving
  seg_map_rotate_640_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "90_" + file))
  # rotate 180 640 raw mask saving
  seg_map_rotate_640_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "180_" + file))
  # rotate 270 640 raw mask saving
  seg_map_rotate_640_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Masks", "270_" + file))

  image_640 = cv2.imread(img_dir + '/' + file, 1)
  image_640 = img_resize(image_640)
  image_640 = Image.fromarray(image_640).convert('RGB')
  os.makedirs("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images")
  # raw 640 image saving
  image_640.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", file.replace('.png','.jpg')))
  # rotate 90 640 raw image saving
  image_640_90 = image_640.rotate(90, expand=1)
  image_640_90.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "90_" +file.replace('.png','.jpg')))
  # rotate 180 640 raw image saving
  image_640_180 = image_640.rotate(180, expand=1)
  image_640_180.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "180_" +file.replace('.png','.jpg')))
  # rotate 270 640 raw image saving
  image_640_270 = image_640.rotate(270, expand=1)
  image_640_270.save(osp.join("./DRAC2022_dataset/Segmentation/Training/A/640/Original_images", "270_" +file.replace('.png','.jpg')))
