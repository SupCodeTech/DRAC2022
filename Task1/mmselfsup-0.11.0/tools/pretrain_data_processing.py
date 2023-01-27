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
  output_list.append("009" + image_name[i][:-4] + ".jpg")
  output_list.append("119" + image_name[i][:-4] + ".jpg")
  output_list.append("909" + image_name[i][:-4] + ".jpg")
  output_list.append("1809" + image_name[i][:-4] + ".jpg")
  output_list.append("2709" + image_name[i][:-4] + ".jpg")
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
  Original_image_480 = cv2.imread(img_dir + '/' + file, 1)
  Original_image_480 = Image.fromarray(Original_image_480).convert('RGB')
  Original_image_480 = Original_image_480.resize((480, 480), Image.ANTIALIAS)
  Data_path_ = './Data/Original_Images/Training_Set'
  Data_path_folder = os.path.exists(Data_path_)
  if not Data_path_folder:
    os.makedirs(Data_path_)
  # raw image saving
  Original_image_480.save(osp.join("./Data/Original_Images/Training_Set", file.replace('.png','.jpg')))
  # Flip horizontal
  image_480_flip_left_right = Original_image_480.transpose(Image.FLIP_LEFT_RIGHT)
  image_480_flip_left_right.save(osp.join("./Data/Original_Images/Training_Set", "009" + file.replace('.png','.jpg')))
  # Flip vertical
  image_480_flip_top_bottom = Original_image_480.transpose(Image.FLIP_TOP_BOTTOM)
  image_480_flip_top_bottom.save(osp.join("./Data/Original_Images/Training_Set", "119" + file.replace('.png','.jpg')))
  # rotate 90 raw image saving
  image_480_90 = Original_image_480.rotate(90, expand=1)
  image_480_90.save(osp.join("./Data/Original_Images/Training_Set", "909" + file.replace('.png','.jpg')))
  # rotate 180 raw image saving
  image_480_180 = Original_image_480.rotate(180, expand=1)
  image_480_180.save(osp.join("./Data/Original_Images/Training_Set", "1809" + file.replace('.png','.jpg')))
  # rotate 270 raw image saving
  image_480_270 = Original_image_480.rotate(270, expand=1)
  image_480_270.save(osp.join("./Data/Original_Images/Training_Set", "2709" + file.replace('.png','.jpg')))
