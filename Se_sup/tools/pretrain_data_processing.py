import cv2
import os.path
import os
import numpy as np
import pandas as pd
import mmcv
from PIL import Image
import os.path as osp

# From the following directories
# ├── Se_sup
# │   ├── C._Diabetic_Retinopathy_Grading
# │   │   ├── 1._Original_Images
# │   │   │   ├── a._Training_Set
# │   │   │   │   ├── 001.png
# │   │   │   │   ├── ...
# │   │   ├── 2._Groundtruths
# │   │   │   ├── a._DRAC2022_Diabetic_Retinopathy_Grading_Training_Labels.csv

# To generate the following directories
# ├── Se_sup
# │   ├── Data
# │   │   ├── Original_Images
# │   │   │   ├── Training_Set
# │   │   │   │   ├── 001.png
# │   │   │   │   ├── ...
# │   │   ├── Pretrained_files.txt

data_pd = pd.read_csv('./C._Diabetic_Retinopathy_Grading/2._Groundtruths/a._DRAC2022_Diabetic_Retinopathy_Grading_Training_Labels.csv')
image_name = list(data_pd['image name'])
DR_grade = list(data_pd['DR grade'])

output_list = []
output_list_ = []

for i in range(611):
  output_list.append(image_name[i][:-4] + ".jpg")
  output_list.append("00_" + image_name[i][:-4] + ".jpg")
  output_list.append("11_" + image_name[i][:-4] + ".jpg")
  output_list.append("90_" + image_name[i][:-4] + ".jpg")
  output_list.append("180_" + image_name[i][:-4] + ".jpg")
  output_list.append("270_" + image_name[i][:-4] + ".jpg")
  for j in range(6):
    output_list_.append(DR_grade[i])
    
Data_path = './Data'
Data_path_folder = os.path.exists(Data_path)
if not Data_path_folder:
  os.makedirs(Data_path)

file = open('./Data/Pretrained_files.txt','w',encoding='utf-8')
for i in range(len(output_list)):
  file.write(str(output_list[i]) + ' ' + str(output_list_[i])  +'\n')
file.close()
img_dir = './C._Diabetic_Retinopathy_Grading/1._Original_Images/a._Training_Set'

for file in mmcv.scandir(img_dir, suffix='.png'):
  Original_image_448 = cv2.imread(img_dir + '/' + file, 1)
  Original_image_448 = Image.fromarray(Original_image_448).convert('RGB')
  Original_image_448 = Original_image_448.resize((448, 448), Image.ANTIALIAS)
  Data_path_ = './Data/Original_Images/Training_Set'
  Data_path_folder = os.path.exists(Data_path_)
  if not Data_path_folder:
    os.makedirs(Data_path_)
  # raw 224 image saving
  Original_image_448.save(osp.join("./Data/Original_Images/Training_Set", file.replace('.png','.jpg')))
  # Flip horizontal
  image_448_flip_left_right = Original_image_448.transpose(Image.FLIP_LEFT_RIGHT)
  image_448_flip_left_right.save(osp.join("./Data/Original_Images/Training_Set", "00_" + file.replace('.png','.jpg')))
  # Flip vertical
  image_448_flip_top_bottom = Original_image_448.transpose(Image.FLIP_TOP_BOTTOM)
  image_448_flip_top_bottom.save(osp.join("./Data/Original_Images/Training_Set", "11_" + file.replace('.png','.jpg')))
  # rotate 90 640 raw image saving
  image_448_90 = Original_image_448.rotate(90, expand=1)
  image_448_90.save(osp.join("./Data/Original_Images/Training_Set", "90_" + file.replace('.png','.jpg')))
  # rotate 180 640 raw image saving
  image_448_180 = Original_image_448.rotate(180, expand=1)
  image_448_180.save(osp.join("./Data/Original_Images/Training_Set", "180_" + file.replace('.png','.jpg')))
  # rotate 270 640 raw image saving
  image_448_270 = Original_image_448.rotate(270, expand=1)
  image_448_270.save(osp.join("./Data/Original_Images/Training_Set", "270_" + file.replace('.png','.jpg')))
