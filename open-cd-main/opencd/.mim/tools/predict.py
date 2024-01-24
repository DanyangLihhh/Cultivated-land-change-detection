import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.nn import CrossEntropyLoss
import os
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import cv2
import numpy as np
# Define your transformation
transform_inputs = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# 假设的模型文件路径和数据集路径
MODEL_PATH = '/root/siton-gpfs-archive/lidanyang/open-cd-main/best_model/changer_ex_r18_512x512_40k_levircd_91.46.pth'

model = torch.load(MODEL_PATH).eval()
celoss = CrossEntropyLoss()
loss_id = []
loss_list = []
# test_path = '/root/siton-gpfs-archive/lidanyang/open-cd-main/not_split_data/test'
# outputs_path = '/root/siton-gpfs-archive/lidanyang/open-cd-main/outputs'

test_path = '/root/siton-gpfs-archive/lidanyang/open-cd-main/data/train'
outputs_path = '/root/siton-gpfs-archive/lidanyang/open-cd-main/outputs/train'

for i in tqdm.tqdm(range(2000)):
    image_pre = transform_inputs(Image.open(test_path+'/pre/image_'+str(i)+'.tif'))[torch.tensor([2, 1, 0]), :, :]
    image_post = transform_inputs(Image.open(test_path+'/post/image_'+str(i)+'.tif'))[torch.tensor([2, 1, 0]), :, :]
    # image_pre = cv2.cvtColor(np.array(image_pre), cv2.COLOR_RGB2BGR)
    # image_post = cv2.cvtColor(np.array(image_post), cv2.COLOR_RGB2BGR)
    output_path = outputs_path + '/image_' +str(i)+ '.png'
    image = torch.cat([torch.tensor(image_pre), torch.tensor(image_post)],dim=0).reshape(1,6,1024,1024)
    output = model(image.cuda())
    output = torch.argmax(output, dim=1).reshape(256,256).cpu().byte().numpy()
    pil_image = Image.fromarray(output, mode='L')
    pil_image.save(output_path)