import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import argparse
import copy
import os
import os.path as osp
import time
import warnings
import numpy as np
from PIL import Image
import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from argparse import ArgumentParser
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes

 
def main():
  parser = ArgumentParser()
  parser.add_argument('load_from_M_config', help='path to store the config of the pretrained model MAE')
  parser.add_argument('load_from_checkpoint_M', help='path to store the checkpoints of the pretrained model MAE')  
  parser.add_argument('load_from_C_config', help='path to store the config of the pretrained model ConvNeXt')
  parser.add_argument('load_from_checkpoint_C', help='path to store the checkpoints of the pretrained model ConvNeXt')
  parser.add_argument('load_from_S_config', help='path to store the config of the pretrained model SegFormer')
  parser.add_argument('load_from_checkpoint_S', help='path to store the checkpoints of the pretrained model SegFormer')
  parser.add_argument('output_data_dir', help='output path of test segmentation results')
  parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
  args = parser.parse_args()
 
  M = init_segmentor(args.load_from_M_config, args.load_from_checkpoint_M,  device='cuda:0')
  C = init_segmentor(args.load_from_C_config, args.load_from_checkpoint_C,  device='cuda:0')
  S = init_segmentor(args.load_from_S_config, args.load_from_checkpoint_S,  device='cuda:0')

  image_size_1 = 1024
  image_size_2 = 1536

  test_data_dir = './DRAC2022_dataset/Test_Data/1536/'
  test_data_dir_ = './DRAC2022_dataset/Test_Data/1024/'

  test_data_dir_1 = test_data_dir + 'Original_Images/'
  test_data_dir_2 = test_data_dir + 'Horizontal_flip/'
  test_data_dir_3 = test_data_dir + 'Vertical_flip/'
  test_data_dir_4 = test_data_dir + 'R90/'
  test_data_dir_5 = test_data_dir + 'R180/'
  test_data_dir_6 = test_data_dir + 'R270/'

  test_data_dir_1_ = test_data_dir_ + 'Original_Images/'
  test_data_dir_2_ = test_data_dir_ + 'Horizontal_flip/'
  test_data_dir_3_ = test_data_dir_ + 'Vertical_flip/'
  test_data_dir_4_ = test_data_dir_ + 'R90/'
  test_data_dir_5_ = test_data_dir_ + 'R180/'
  test_data_dir_6_ = test_data_dir_ + 'R270/'
  
  Intraretinal_microvascular_abnormals_Mask_path = output_data_dir + '/Intraretinal_microvascular_abnormals_Mask'
  Intraretinal_microvascular_abnormals_Mask_path_folder = os.path.exists(Intraretinal_microvascular_abnormals_Mask_path)
  if not Intraretinal_microvascular_abnormals_Mask_path_folder:
    os.makedirs(Intraretinal_microvascular_abnormals_Mask_path)

  Neovascularization_Mask_path = output_data_dir + '/Neovascularization_Mask'
  Neovascularization_Mask_path_folder = os.path.exists(Neovascularization_Mask_path)
  if not Neovascularization_Mask_path_folder:
    os.makedirs(Neovascularization_Mask_path)

  for file in mmcv.scandir(test_data_dir_1, suffix='.png'):
    
    image_file_1 = os.path.join(test_data_dir_1 + file)

    Intraretinal_microvascular_abnormals_Mask_1 = np.zeros(2359296).reshape((image_size_2, image_size_2)) 
    Neovascularization_Mask_1 = np.zeros(2359296).reshape((image_size_2, image_size_2))

    Intraretinal_microvascular_abnormals_Mask_1_ = np.zeros(1048576).reshape((image_size_1, image_size_1)) 
    Neovascularization_Mask_1_ = np.zeros(1048576).reshape((image_size_1, image_size_1))


    result_1 = inference_segmentor(M, image_file_1)
    result_2 = inference_segmentor(C, image_file_1)
    result_3 = inference_segmentor(S, image_file_1)
    result_3_ = inference_segmentor(S, image_file_1_)

    for i in range(image_size_2):
      for j in range(image_size_2):
        if (result_1[0][i][j] == 1) or (result_2[0][i][j] == 1) or (result_3[0][i][j] == 1):
          Intraretinal_microvascular_abnormals_Mask_1[i][j] = 25
        elif (result_1[0][i][j] == 2) or (result_2[0][i][j] == 2) or (result_3[0][i][j] == 2):
          Neovascularization_Mask_1[i][j] = 25
        elif (result_1[0][i][j] == 3) or (result_2[0][i][j] == 3) or (result_3[0][i][j] == 3):
          Intraretinal_microvascular_abnormals_Mask_1[i][j] = 25
          Neovascularization_Mask_1[i][j] = 25

    for i in range(image_size_1):
      for j in range(image_size_1):
        if result_3_[0][i][j] == 1:
          Intraretinal_microvascular_abnormals_Mask_1_[i][j] = 25
        elif result_3_[0][i][j] == 2:
          Neovascularization_Mask_1_[i][j] = 25
        elif result_3_[0][i][j] == 3:
          Intraretinal_microvascular_abnormals_Mask_1_[i][j] = 25
          Neovascularization_Mask_1_[i][j] = 25

    image_file_2 = os.path.join(test_data_dir_2 + file)

    Intraretinal_microvascular_abnormals_Mask_2 = np.zeros(2359296).reshape((image_size_2, image_size_2)) 
    Neovascularization_Mask_2 = np.zeros(2359296).reshape((image_size_2, image_size_2))

    Intraretinal_microvascular_abnormals_Mask_2_ = np.zeros(1048576).reshape((image_size_1, image_size_1)) 
    Neovascularization_Mask_2_ = np.zeros(1048576).reshape((image_size_1, image_size_1))

    result_1 = inference_segmentor(M, image_file_2)
    result_2 = inference_segmentor(C, image_file_2)
    result_3 = inference_segmentor(S, image_file_2)
    result_3_ = inference_segmentor(S, image_file_2_)
    
    for i in range(image_size_2):
      for j in range(image_size_2):
        if (result_1[0][i][j] == 1) or (result_2[0][i][j] == 1) or (result_3[0][i][j] == 1):
          Intraretinal_microvascular_abnormals_Mask_2[i][j] = 25
        elif (result_1[0][i][j] == 2) or (result_2[0][i][j] == 2) or (result_3[0][i][j] == 2):
          Neovascularization_Mask_2[i][j] = 25
        elif (result_1[0][i][j] == 3) or (result_2[0][i][j] == 3) or (result_3[0][i][j] == 3):
          Intraretinal_microvascular_abnormals_Mask_2[i][j] = 25
          Neovascularization_Mask_2[i][j] = 25

    for i in range(image_size_1):
      for j in range(image_size_1):
        if result_3_[0][i][j] == 1:
          Intraretinal_microvascular_abnormals_Mask_2_[i][j] = 25
        elif result_3_[0][i][j] == 2:
          Neovascularization_Mask_2_[i][j] = 25
        elif result_3_[0][i][j] == 3:
          Intraretinal_microvascular_abnormals_Mask_2_[i][j] = 25
          Neovascularization_Mask_2_[i][j] = 25  

    image_file_3 = os.path.join(test_data_dir_3 + file)

    Intraretinal_microvascular_abnormals_Mask_3 = np.zeros(2359296).reshape((image_size_2, image_size_2)) 
    Neovascularization_Mask_3 = np.zeros(2359296).reshape((image_size_2, image_size_2))

    Intraretinal_microvascular_abnormals_Mask_3_ = np.zeros(1048576).reshape((image_size_1, image_size_1)) 
    Neovascularization_Mask_3_ = np.zeros(1048576).reshape((image_size_1, image_size_1))

    result_1 = inference_segmentor(M, image_file_3)
    result_2 = inference_segmentor(C, image_file_3)
    result_3 = inference_segmentor(S, image_file_3)
    result_3_ = inference_segmentor(S, image_file_3_)
    
    for i in range(image_size_2):
      for j in range(image_size_2):
        if (result_1[0][i][j] == 1) or (result_2[0][i][j] == 1) or (result_3[0][i][j] == 1):
          Intraretinal_microvascular_abnormals_Mask_3[i][j] = 25
        elif (result_1[0][i][j] == 2) or (result_2[0][i][j] == 2) or (result_3[0][i][j] == 2):
          Neovascularization_Mask_3[i][j] = 25
        elif (result_1[0][i][j] == 3) or (result_2[0][i][j] == 3) or (result_3[0][i][j] == 3):
          Intraretinal_microvascular_abnormals_Mask_3[i][j] = 25
          Neovascularization_Mask_3[i][j] = 25

    for i in range(image_size_1):
      for j in range(image_size_1):
        if result_3_[0][i][j] == 1:
          Intraretinal_microvascular_abnormals_Mask_3_[i][j] = 25
        elif result_3_[0][i][j] == 2:
          Neovascularization_Mask_3_[i][j] = 25
        elif result_3_[0][i][j] == 3:
          Intraretinal_microvascular_abnormals_Mask_3_[i][j] = 25
          Neovascularization_Mask_3_[i][j] = 25  

    image_file_4 = os.path.join(test_data_dir_4 + file)

    Intraretinal_microvascular_abnormals_Mask_4 = np.zeros(2359296).reshape((image_size_2, image_size_2)) 
    Neovascularization_Mask_4 = np.zeros(2359296).reshape((image_size_2, image_size_2))

    Intraretinal_microvascular_abnormals_Mask_4_ = np.zeros(1048576).reshape((image_size_1, image_size_1)) 
    Neovascularization_Mask_4_ = np.zeros(1048576).reshape((image_size_1, image_size_1))

    result_1 = inference_segmentor(M, image_file_4)
    result_2 = inference_segmentor(C, image_file_4)
    result_3 = inference_segmentor(S, image_file_4)
    result_3_ = inference_segmentor(S, image_file_4_)

    for i in range(image_size_2):
      for j in range(image_size_2):
        if (result_1[0][i][j] == 1) or (result_2[0][i][j] == 1) or (result_3[0][i][j] == 1):
          Intraretinal_microvascular_abnormals_Mask_4[i][j] = 25
        elif (result_1[0][i][j] == 2) or (result_2[0][i][j] == 2) or (result_3[0][i][j] == 2):
          Neovascularization_Mask_4[i][j] = 25
        elif (result_1[0][i][j] == 3) or (result_2[0][i][j] == 3) or (result_3[0][i][j] == 3):
          Intraretinal_microvascular_abnormals_Mask_4[i][j] = 25
          Neovascularization_Mask_4[i][j] = 25

    for i in range(image_size_1):
      for j in range(image_size_1):
        if result_3_[0][i][j] == 1:
          Intraretinal_microvascular_abnormals_Mask_4_[i][j] = 25
        elif result_3_[0][i][j] == 2:
          Neovascularization_Mask_4_[i][j] = 25
        elif result_3_[0][i][j] == 3:
          Intraretinal_microvascular_abnormals_Mask_4_[i][j] = 25
          Neovascularization_Mask_4_[i][j] = 25  

    image_file_5 = os.path.join(test_data_dir_5 + file)

    Intraretinal_microvascular_abnormals_Mask_5 = np.zeros(2359296).reshape((image_size_2, image_size_2)) 
    Neovascularization_Mask_5 = np.zeros(2359296).reshape((image_size_2, image_size_2))

    Intraretinal_microvascular_abnormals_Mask_5_ = np.zeros(1048576).reshape((image_size_1, image_size_1)) 
    Neovascularization_Mask_5_ = np.zeros(1048576).reshape((image_size_1, image_size_1))
    
    result_1 = inference_segmentor(M, image_file_5)
    result_2 = inference_segmentor(C, image_file_5)
    result_3 = inference_segmentor(S, image_file_5)
    result_3_ = inference_segmentor(S, image_file_5_)
    
    for i in range(image_size_2):
      for j in range(image_size_2):
        if (result_1[0][i][j] == 1) or (result_2[0][i][j] == 1) or (result_3[0][i][j] == 1):
          Intraretinal_microvascular_abnormals_Mask_5[i][j] = 25
        elif (result_1[0][i][j] == 2) or (result_2[0][i][j] == 2) or (result_3[0][i][j] == 2):
          Neovascularization_Mask_5[i][j] = 25
        elif (result_1[0][i][j] == 3) or (result_2[0][i][j] == 3) or (result_3[0][i][j] == 3):
          Intraretinal_microvascular_abnormals_Mask_5[i][j] = 25
          Neovascularization_Mask_5[i][j] = 25

    for i in range(image_size_1):
      for j in range(image_size_1):
        if result_3_[0][i][j] == 1:
          Intraretinal_microvascular_abnormals_Mask_5_[i][j] = 25
        elif result_3_[0][i][j] == 2:
          Neovascularization_Mask_5_[i][j] = 25
        elif result_3_[0][i][j] == 3:
          Intraretinal_microvascular_abnormals_Mask_5_[i][j] = 25
          Neovascularization_Mask_5_[i][j] = 25 

    image_file_6 = os.path.join(test_data_dir_6 + file)

    Intraretinal_microvascular_abnormals_Mask_6 = np.zeros(2359296).reshape((image_size_2, image_size_2)) 
    Neovascularization_Mask_6 = np.zeros(2359296).reshape((image_size_2, image_size_2))

    Intraretinal_microvascular_abnormals_Mask_6_ = np.zeros(1048576).reshape((image_size_1, image_size_1)) 
    Neovascularization_Mask_6_ = np.zeros(1048576).reshape((image_size_1, image_size_1))

    result_1 = inference_segmentor(M, image_file_6)
    result_2 = inference_segmentor(C, image_file_6)
    result_3 = inference_segmentor(S, image_file_6)
    result_3_ = inference_segmentor(S, image_file_6_)
    
    for i in range(image_size_2):
      for j in range(image_size_2):
        if (result_1[0][i][j] == 1) or (result_2[0][i][j] == 1) or (result_3[0][i][j] == 1):
          Intraretinal_microvascular_abnormals_Mask_6[i][j] = 25
        elif (result_1[0][i][j] == 2) or (result_2[0][i][j] == 2) or (result_3[0][i][j] == 2):
          Neovascularization_Mask_6[i][j] = 25
        elif (result_1[0][i][j] == 3) or (result_2[0][i][j] == 3) or (result_3[0][i][j] == 3):
          Intraretinal_microvascular_abnormals_Mask_6[i][j] = 25
          Neovascularization_Mask_6[i][j] = 25

    for i in range(image_size_1):
      for j in range(image_size_1):
        if result_3_[0][i][j] == 1:
          Intraretinal_microvascular_abnormals_Mask_6_[i][j] = 25
        elif result_3_[0][i][j] == 2:
          Neovascularization_Mask_6_[i][j] = 25
        elif result_3_[0][i][j] == 3:
          Intraretinal_microvascular_abnormals_Mask_6_[i][j] = 25
          Neovascularization_Mask_6_[i][j] = 25

    # Intraretinal_microvascular_abnormals_Mask
    # 1536 position recovery
    Intraretinal_microvascular_abnormals_Mask_1 = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_1).convert('P')
    Intraretinal_microvascular_abnormals_Mask_1 = Intraretinal_microvascular_abnormals_Mask_1.resize((1024, 1024), Image.ANTIALIAS)
    Intraretinal_microvascular_abnormals_Mask_1 = np.asarray(Intraretinal_microvascular_abnormals_Mask_1)

    Intraretinal_microvascular_abnormals_Mask_2 = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_2).convert('P')
    Intraretinal_microvascular_abnormals_Mask_2 = Intraretinal_microvascular_abnormals_Mask_2.transpose(Image.FLIP_LEFT_RIGHT)
    Intraretinal_microvascular_abnormals_Mask_2 = Intraretinal_microvascular_abnormals_Mask_2.resize((1024, 1024), Image.ANTIALIAS)
    Intraretinal_microvascular_abnormals_Mask_2 = np.asarray(Intraretinal_microvascular_abnormals_Mask_2)

    Intraretinal_microvascular_abnormals_Mask_3 = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_3).convert('P')
    Intraretinal_microvascular_abnormals_Mask_3 = Intraretinal_microvascular_abnormals_Mask_3.transpose(Image.FLIP_TOP_BOTTOM)
    Intraretinal_microvascular_abnormals_Mask_3 = Intraretinal_microvascular_abnormals_Mask_3.resize((1024, 1024), Image.ANTIALIAS)
    Intraretinal_microvascular_abnormals_Mask_3 = np.asarray(Intraretinal_microvascular_abnormals_Mask_3)

    Intraretinal_microvascular_abnormals_Mask_4 = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_4).convert('P')
    Intraretinal_microvascular_abnormals_Mask_4 = Intraretinal_microvascular_abnormals_Mask_4.rotate(270, expand=1)
    Intraretinal_microvascular_abnormals_Mask_4 = Intraretinal_microvascular_abnormals_Mask_4.resize((1024, 1024), Image.ANTIALIAS)
    Intraretinal_microvascular_abnormals_Mask_4 = np.asarray(Intraretinal_microvascular_abnormals_Mask_4)

    Intraretinal_microvascular_abnormals_Mask_5 = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_5).convert('P')
    Intraretinal_microvascular_abnormals_Mask_5 = Intraretinal_microvascular_abnormals_Mask_5.rotate(180, expand=1)
    Intraretinal_microvascular_abnormals_Mask_5 = Intraretinal_microvascular_abnormals_Mask_5.resize((1024, 1024), Image.ANTIALIAS)
    Intraretinal_microvascular_abnormals_Mask_5 = np.asarray(Intraretinal_microvascular_abnormals_Mask_5)

    Intraretinal_microvascular_abnormals_Mask_6 = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_6).convert('P')
    Intraretinal_microvascular_abnormals_Mask_6 = Intraretinal_microvascular_abnormals_Mask_6.rotate(90, expand=1)
    Intraretinal_microvascular_abnormals_Mask_6 = Intraretinal_microvascular_abnormals_Mask_6.resize((1024, 1024), Image.ANTIALIAS)
    Intraretinal_microvascular_abnormals_Mask_6 = np.asarray(Intraretinal_microvascular_abnormals_Mask_6)

    # 1024 position recovery
    Intraretinal_microvascular_abnormals_Mask_1_ = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_1_).convert('P')
    Intraretinal_microvascular_abnormals_Mask_1_ = np.asarray(Intraretinal_microvascular_abnormals_Mask_1_)

    Intraretinal_microvascular_abnormals_Mask_2_ = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_2_).convert('P')
    Intraretinal_microvascular_abnormals_Mask_2_ = Intraretinal_microvascular_abnormals_Mask_2_.transpose(Image.FLIP_LEFT_RIGHT)
    Intraretinal_microvascular_abnormals_Mask_2_ = np.asarray(Intraretinal_microvascular_abnormals_Mask_2_)

    Intraretinal_microvascular_abnormals_Mask_3_ = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_3_).convert('P')
    Intraretinal_microvascular_abnormals_Mask_3_ = Intraretinal_microvascular_abnormals_Mask_3_.transpose(Image.FLIP_TOP_BOTTOM)
    Intraretinal_microvascular_abnormals_Mask_3_ = np.asarray(Intraretinal_microvascular_abnormals_Mask_3_)

    Intraretinal_microvascular_abnormals_Mask_4_ = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_4_).convert('P')
    Intraretinal_microvascular_abnormals_Mask_4_ = Intraretinal_microvascular_abnormals_Mask_4_.rotate(270, expand=1)
    Intraretinal_microvascular_abnormals_Mask_4_ = np.asarray(Intraretinal_microvascular_abnormals_Mask_4_)

    Intraretinal_microvascular_abnormals_Mask_5_ = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_5_).convert('P')
    Intraretinal_microvascular_abnormals_Mask_5_ = Intraretinal_microvascular_abnormals_Mask_5_.rotate(180, expand=1)
    Intraretinal_microvascular_abnormals_Mask_5_ = np.asarray(Intraretinal_microvascular_abnormals_Mask_5_)

    Intraretinal_microvascular_abnormals_Mask_6_ = Image.fromarray(Intraretinal_microvascular_abnormals_Mask_6_).convert('P')
    Intraretinal_microvascular_abnormals_Mask_6_ = Intraretinal_microvascular_abnormals_Mask_6_.rotate(90, expand=1)
    Intraretinal_microvascular_abnormals_Mask_6_ = np.asarray(Intraretinal_microvascular_abnormals_Mask_6_)

    Intraretinal_microvascular_abnormals_Mask = Intraretinal_microvascular_abnormals_Mask_1 \
    + Intraretinal_microvascular_abnormals_Mask_2 + Intraretinal_microvascular_abnormals_Mask_3 \
    + Intraretinal_microvascular_abnormals_Mask_4 + Intraretinal_microvascular_abnormals_Mask_5 \
    + Intraretinal_microvascular_abnormals_Mask_6 + Intraretinal_microvascular_abnormals_Mask_1_ \
    + Intraretinal_microvascular_abnormals_Mask_2_ + Intraretinal_microvascular_abnormals_Mask_3_ \
    + Intraretinal_microvascular_abnormals_Mask_4_ + Intraretinal_microvascular_abnormals_Mask_5_ \
    + Intraretinal_microvascular_abnormals_Mask_6_

    for i in range(1024):
      for j in range(1024):
        if Intraretinal_microvascular_abnormals_Mask[i][j] > 0:
          Intraretinal_microvascular_abnormals_Mask[i][j] = 255

    Intraretinal_microvascular_abnormals_Mask = Image.fromarray(Intraretinal_microvascular_abnormals_Mask).convert('P')
    Intraretinal_microvascular_abnormals_Mask.save(osp.join(Intraretinal_microvascular_abnormals_Mask_path, file))

    # Neovascularization_Mask
    # 1536 position recovery
    Neovascularization_Mask_1 = Image.fromarray(Neovascularization_Mask_1).convert('P')
    Neovascularization_Mask_1 = Neovascularization_Mask_1.resize((1024, 1024), Image.ANTIALIAS)
    Neovascularization_Mask_1 = np.asarray(Neovascularization_Mask_1)

    Neovascularization_Mask_2 = Image.fromarray(Neovascularization_Mask_2).convert('P')
    Neovascularization_Mask_2 = Neovascularization_Mask_2.transpose(Image.FLIP_LEFT_RIGHT)
    Neovascularization_Mask_2 = Neovascularization_Mask_2.resize((1024, 1024), Image.ANTIALIAS)
    Neovascularization_Mask_2 = np.asarray(Neovascularization_Mask_2)

    Neovascularization_Mask_3 = Image.fromarray(Neovascularization_Mask_3).convert('P')
    Neovascularization_Mask_3 = Neovascularization_Mask_3.transpose(Image.FLIP_TOP_BOTTOM)
    Neovascularization_Mask_3 = Neovascularization_Mask_3.resize((1024, 1024), Image.ANTIALIAS)
    Neovascularization_Mask_3 = np.asarray(Neovascularization_Mask_3)

    Neovascularization_Mask_4 = Image.fromarray(Neovascularization_Mask_4).convert('P')
    Neovascularization_Mask_4 = Neovascularization_Mask_4.rotate(270, expand=1)
    Neovascularization_Mask_4 = Neovascularization_Mask_4.resize((1024, 1024), Image.ANTIALIAS)
    Neovascularization_Mask_4 = np.asarray(Neovascularization_Mask_4)

    Neovascularization_Mask_5 = Image.fromarray(Neovascularization_Mask_5).convert('P')
    Neovascularization_Mask_5 = Neovascularization_Mask_5.rotate(180, expand=1)
    Neovascularization_Mask_5 = Neovascularization_Mask_5.resize((1024, 1024), Image.ANTIALIAS)
    Neovascularization_Mask_5 = np.asarray(Neovascularization_Mask_5)

    Neovascularization_Mask_6 = Image.fromarray(Neovascularization_Mask_6).convert('P')
    Neovascularization_Mask_6 = Neovascularization_Mask_6.rotate(90, expand=1)
    Neovascularization_Mask_6 = Neovascularization_Mask_6.resize((1024, 1024), Image.ANTIALIAS)
    Neovascularization_Mask_6 = np.asarray(Neovascularization_Mask_6)

    # 1024 position recovery
    Neovascularization_Mask_1_ = Image.fromarray(Neovascularization_Mask_1_).convert('P')
    Neovascularization_Mask_1_ = np.asarray(Neovascularization_Mask_1_)

    Neovascularization_Mask_2_ = Image.fromarray(Neovascularization_Mask_2_).convert('P')
    Neovascularization_Mask_2_ = Neovascularization_Mask_2_.transpose(Image.FLIP_LEFT_RIGHT)
    Neovascularization_Mask_2_ = np.asarray(Neovascularization_Mask_2_)

    Neovascularization_Mask_3_ = Image.fromarray(Neovascularization_Mask_3_).convert('P')
    Neovascularization_Mask_3_ = Neovascularization_Mask_3_.transpose(Image.FLIP_TOP_BOTTOM)
    Neovascularization_Mask_3_ = np.asarray(Neovascularization_Mask_3_)

    Neovascularization_Mask_4_ = Image.fromarray(Neovascularization_Mask_4_).convert('P')
    Neovascularization_Mask_4_ = Neovascularization_Mask_4_.rotate(270, expand=1)
    Neovascularization_Mask_4_ = np.asarray(Neovascularization_Mask_4_)

    Neovascularization_Mask_5_ = Image.fromarray(Neovascularization_Mask_5_).convert('P')
    Neovascularization_Mask_5_ = Neovascularization_Mask_5_.rotate(180, expand=1)
    Neovascularization_Mask_5_ = np.asarray(Neovascularization_Mask_5_)

    Neovascularization_Mask_6_ = Image.fromarray(Neovascularization_Mask_6_).convert('P')
    Neovascularization_Mask_6_ = Neovascularization_Mask_6_.rotate(90, expand=1)
    Neovascularization_Mask_6_ = np.asarray(Neovascularization_Mask_6_)

    Neovascularization_Mask = Neovascularization_Mask_1 \
    + Neovascularization_Mask_2 + Neovascularization_Mask_3 \
    + Neovascularization_Mask_4 + Neovascularization_Mask_5 \
    + Neovascularization_Mask_6 + Neovascularization_Mask_1_ \
    + Neovascularization_Mask_2_ + Neovascularization_Mask_3_ \
    + Neovascularization_Mask_4_ + Neovascularization_Mask_5_ \
    + Neovascularization_Mask_6_

    for i in range(1024):
      for j in range(1024):
        if Neovascularization_Mask[i][j] > 0:
          Neovascularization_Mask[i][j] = 255

    Neovascularization_Mask = Image.fromarray(Neovascularization_Mask).convert('P')
    Neovascularization_Mask.save(osp.join(Neovascularization_Mask_path, file))

if __name__ == '__main__':
    main()
