import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.nn import CrossEntropyLoss
import os
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
class CustomDataset(Dataset):
    def __init__(self, pre_folder, post_folder, label_folder, transform_inputs=None, transform_labels=None):
        self.pre_folder = pre_folder
        self.post_folder = post_folder
        self.label_folder = label_folder
        self.transform_inputs = transform_inputs
        self.transform_labels = transform_labels
        self.file_list = sorted(os.listdir(self.pre_folder))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pre_img_path = os.path.join(self.pre_folder, self.file_list[idx])
        post_img_path = os.path.join(self.post_folder, self.file_list[idx])
        label_img_path = os.path.join(self.label_folder, self.file_list[idx].split('.')[0]+'.png')
        
        pre_image = Image.open(pre_img_path).convert('RGB')
        post_image = Image.open(post_img_path).convert('RGB')
        label_image = Image.open(label_img_path).convert('L')
        
        pre_image = self.transform_inputs(pre_image)
        post_image = self.transform_inputs(post_image)
        label_image = self.transform_labels(label_image)

        return pre_image, post_image, label_image

# Define your transformation
transform_inputs = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

transform_labels = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 假设的模型文件路径和数据集路径
MODEL_PATH = '/root/siton-gpfs-archive/lidanyang/open-cd-main/best_model/changer_ex_s50_512x512_40k_levircd_89.48.pth'

# 加载模型
model = torch.load(MODEL_PATH)
model.eval()
pre_folder_path = '/root/siton-gpfs-archive/lidanyang/open-cd-main/not_split_data/train/pre'
post_folder_path = '/root/siton-gpfs-archive/lidanyang/open-cd-main/not_split_data/train/post'
label_folder_path = '/root/siton-gpfs-archive/lidanyang/open-cd-main/not_split_data/train/label'
custom_dataset = CustomDataset(pre_folder=pre_folder_path, post_folder=post_folder_path, label_folder=label_folder_path, transform_inputs=transform_inputs, transform_labels=transform_labels)

# # 加载数据集
# dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

# # 初始化损失函数
# criterion = CrossEntropyLoss()

# # 存储损失值和图片名
# losses = []
# image_id = 0
# # 对数据集中的每张图片进行预测并计算损失
# for pre, post, labels in tqdm.tqdm(dataloader):
#     image=torch.cat([pre,post],dim=1)
#     outputs = model(image.cuda())
#     labels = torch.tensor(labels, dtype=torch.long)
#     loss = criterion(outputs.cuda(), labels.reshape(1,256,256).cuda())
#     losses.append((loss.item(), image_id))
#     image_id += 1

# # 根据损失值排序并获取前100张图片
# top_100_losses = sorted(losses, key=lambda x: x[0], reverse=True)[:100]

# # 打印或处理前100张图片的信息
# for loss, image in top_100_losses:
#     print(f"Loss: {loss}, Image: {image}")

# exa:
model = torch.load(MODEL_PATH).eval()
celoss = CrossEntropyLoss()
loss_id = []
loss_list = []
for i in tqdm.tqdm(range(2000)):
    image_pre = transform_inputs(Image.open('/root/siton-gpfs-archive/lidanyang/open-cd-main/not_split_data/test/pre/image_'+str(i)+'.tif'))
    image_post = transform_inputs(Image.open('/root/siton-gpfs-archive/lidanyang/open-cd-main/not_split_data/test/post/image_'+str(i)+'.tif'))
    # image_label = transform_labels(Image.open('/root/siton-gpfs-archive/lidanyang/open-cd-main/not_split_data/train/label/image_'+str(i)+'.png'))
    image = torch.cat([image_pre, image_post],dim=0).reshape(1,6,1024,1024)
    outputs = model(image.cuda())
    # loss = celoss(outputs.cuda(), torch.tensor(image_label.reshape(1,256,256).cuda(), dtype=torch.long))
    # loss_id.append((loss.item(),i))
    # loss_list.append(loss.item())
    image = Image.fromarray(matrix)
    plt.imshow(image,'gray')
    plt.savefig(output_path)
    plt.show()
# top_100_loss_id = sorted(loss_id, key=lambda x: x[0], reverse=True)[:100]
# loss_mean = sum(loss_list)/
# print(top_100_loss_id)
# print(loss_mean)
