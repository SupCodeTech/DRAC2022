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
  
  parser.add_argument('load_from_C_config', help='path to store the config of the pretrained model ConvNeXt')
  parser.add_argument('load_from_checkpoint_C', help='path to store the checkpoints of the pretrained model ConvNeXt')
  parser.add_argument('output_data_dir', help='output path of test segmentation results')
  parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
  args = parser.parse_args()
  
  C = init_segmentor(args.load_from_checkpoint_C_config, args.load_from_checkpoint_C,  device='cuda:0')

  image_size = 1536

  test_data_dir = './DRAC2022_dataset/Test_Data/1536/'

  test_data_dir_1 = test_data_dir + 'Original_Images/'
  
  Nonperfusion_Areas_Mask_path = args.output_data_dir + '/Nonperfusion_Areas'
  Nonperfusion_Areas_Mask_path_folder = os.path.exists(Nonperfusion_Areas_Mask_path)
  if not Nonperfusion_Areas_Mask_path_folder:
    os.makedirs(Nonperfusion_Areas_Mask_path)


  for file in mmcv.scandir(test_data_dir_1, suffix='.png'):
    
    image_file_1 = os.path.join(test_data_dir_1 + file)

    Nonperfusion_Areas_Mask = np.zeros(2359296).reshape((image_size, image_size)) 

    result_ = inference_segmentor(C, image_file_1)

    for i in range(image_size):
      for j in range(image_size):
        if result_[0][i][j] > 0:
          Nonperfusion_Areas_Mask[i][j] = 255

    # Intraretinal_microvascular_abnormals_Mask
    # 1536 position recovery
    Nonperfusion_Areas_Mask = Image.fromarray(Nonperfusion_Areas_Mask).convert('P')
    Nonperfusion_Areas_Mask = Nonperfusion_Areas_Mask.resize((1024, 1024), Image.ANTIALIAS)
    Nonperfusion_Areas_Mask = np.asarray(Nonperfusion_Areas_Mask)

    for i in range(1024):
      for j in range(1024):
        if Nonperfusion_Areas_Mask[i][j] > 0:
          Nonperfusion_Areas_Mask[i][j] = 255

    Nonperfusion_Areas_Mask = Image.fromarray(Nonperfusion_Areas_Mask).convert('P')
    Nonperfusion_Areas_Mask.save(osp.join(Nonperfusion_Areas_Mask_path, file))

if __name__ == '__main__':
    main()
