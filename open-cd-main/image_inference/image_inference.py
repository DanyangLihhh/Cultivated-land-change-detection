from opencd.apis import OpenCDInferencer
import os
from PIL import Image

# Load models into memory
inferencer = OpenCDInferencer(model='configs/changer/changer_ex_r18_256x256_40k_jilinone.py',
                               weights='best_score.pth',
                               classes=('0', '1', '2', '3', '4', '5', '6', '7', '8'), 
                               palette=[[255, 255, 0], [128, 128, 1], [130, 87, 2], [255, 0, 3], [0, 0, 4],[64,128,5],[64,128,6],[24,24,7],[100,200,8]])
# Inference
data_path = 'data/test/'
out_dir = 'OUTPUT_PATH'
data_pre_path = data_path + 'pre'
data_post_path = data_path + 'post'
all_imgs_list = []

for img_name in os.listdir(data_pre_path):
    img_pre_path = data_pre_path + "/" + img_name
    img_post_path = data_post_path + "/" + img_name
    all_imgs_list.append([img_pre_path, img_post_path])

inferencer(all_imgs_list, show=False, out_dir=out_dir)

# Generate prediction results to meet submission requirements

print("Inference finished. Generating results...")

os.mkdir(out_dir+'/results')

for img_name in os.listdir(out_dir+'/vis'):
    vis_image_path = os.path.join(out_dir, 'vis', img_name)
    pred_output_path = os.path.join(out_dir, 'results', img_name)
    image = Image.open(vis_image_path)
    gray_image = image.split()[2]
    gray_image.save(pred_output_path)