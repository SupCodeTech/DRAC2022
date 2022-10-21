import torch
import collections
import os.path as osp
from collections import OrderedDict
import mmcv
import torch
from mmcv.runner import CheckpointLoader

def convert_beit(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('patch_embed'):
            new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
            new_ckpt[new_key] = v
        if k.startswith('blocks'):
            new_key = k.replace('blocks', 'layers')
            if 'norm' in new_key:
                new_key = new_key.replace('norm', 'ln')
            elif 'mlp.fc1' in new_key:
                new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in new_key:
                new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
            new_ckpt[new_key] = v
        else:
            new_key = k
            new_ckpt[new_key] = v
    return new_ckpt
ck = torch.load("./work_dirs/mae/epoch_16000.pth", map_location=torch.device('cpu'))

def main():
    checkpoint = CheckpointLoader.load_checkpoint("https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth", map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_beit(state_dict)
    mmcv.mkdir_or_exist(osp.dirname("./pretrain/mae_pretrain_vit_base_mmcls.pth"))
    torch.save(weight, "./pretrain/mae_pretrain_vit_base_mmcls.pth")
    output_dict = dict()
    net=torch.load('./pretrain/mae_pretrain_vit_base_mmcls.pth')
    for k in net.items():
      has_backbone = False
      for key, value in ck['state_dict'].items():
        if key.startswith('backbone') and key == k[0] and value.shape == k[1].shape:
          output_dict[key[9:]] = value
          has_backbone = True
      if has_backbone == False:
        output_dict[k[0]] = k[1]
    torch.save(output_dict, "./work_dirs/mae/pretrain_backbone_16k.pth")
if __name__ == '__main__':
    main()
