from PIL import Image
import os
import tqdm
input_folder = r'E:\all things\change_detection\tmp_infer_resnest91.81\tmp_infer_resnest91.81\vis_data\vis_image'
output_folder = r'E:\all things\change_detection\tmp_infer_resnest91.81\tmp_infer_resnest91.81\vis_data\vis_image_gray'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的每张图片
for i in tqdm.tqdm(range(6001)):
    if os.path.exists(os.path.join(input_folder, f'image_{i}.png')):
        image_path = os.path.join(input_folder, f'image_{i}.png')
        output_path = os.path.join(output_folder, f'image_{i}.png')

        # 打开图像
        image = Image.open(image_path)

        # 将图像转为灰度
        gray_image = image.split()[2]  # 通道索引从0开始，因此第三个通道索引是2

        # 保存灰度图像
        gray_image.save(output_path)
    else:
        continue

print("处理完成")
