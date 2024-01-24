import torch
from mmseg.apis import inference_model, init_model

config_file = '/root/siton-gpfs-archive/lidanyang/open-cd-main/configs/changer/changer_ex_r18_512x512_40k_levircd.py'
checkpoint_file = '/root/siton-gpfs-archive/lidanyang/open-cd-main/work_dirs/changer_ex_r18_512x512_40k_levircd/best_mIoU_iter_36000.pth'

# 初始化分割器
model = init_model(config_file, checkpoint_file, device='cuda:0')

torch.save(model, '/root/siton-gpfs-archive/lidanyang/open-cd-main/best_model/changer_ex_r18_512x512_40k_levircd_91.46.pth')