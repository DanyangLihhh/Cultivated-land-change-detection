# 参赛团队

Secret;Weapon

# 引言

本次比赛的代码基于opencd和mmsegmentation框架实现。

本次比赛的网络结构等技术方案详见我们提交成果中的“技术方案.pdf”。这里我们清楚的描述了对数据的预处理、整体的网络框架、每个模块的参数细节、实验详情等。

对于本次比赛要求提交的best_score.pth和requirements均被放置在open-cd-main文件夹中。

# 启动之前

本次比赛代码的环境均在linux系统上进行配置。如果环境配置过程中遇到问题，请及时联系我们，我们非常乐意远程协助环境的配置。

在运行代码之前，请确保您已完成以下准备工作：

## 配置opencd框架所需要的包库

使用pycharm或vscode等编译器，将"open-cd-main"作为工程文件夹打开。

### 依赖

在本节中，我们将演示如何用PyTorch准备一个环境。

它需要 Python 3.6 以上，CUDA 9.2 以上和 PyTorch 1.3 以上。

```
如果您对PyTorch有经验并且已经安装了它，请跳到下一节。否则，您可以按照以下步骤进行准备。
```

**第一步** 从[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**第二步** 创建并激活一个 conda 环境。

```shell
conda create --name changedet python=3.8 -y
conda activate changedet
```

**第三步** 按照[官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch。

在 GPU 平台上：

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 安装opencd所需要的包库

```
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
mim install "mmpretrain>=1.0.0rc7"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install ftfy
pip install timm
```
```
cd open-cd-main
pip install -v -e .
```

## 配置mmsegmentation所需环境

参考 [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) 完成mmsegmentation的环境配置。

注：本节仅在上一节环境配置失败的情况下使用，若上一节环境配置成功可跳过此节。如果环境配置出现问题，请尽快联系我们。我们非常乐意远程为本此比赛的代码评测配置环境。

## 使用本框架

1、存放数据

请在data文件夹中存放好用于模型训练与测试的三个数据文件夹：train、val和test。由于本次比赛并未提供val数据，我们选择随机从train中取出500对图片作为我们的val数据。如果仅为测试我们的代码框架，您可以选择将第0-499对图片作为val。

```
|---data
    |---train
    |---val
    |---test
```

2、模型训练

请在终端中激活配置的环境，并进入本项目的文件夹，输入以下命令：

```
python ./tools/train.py ./configs/changer/changer_ex_r18_256x256_40k_jilinone.py --work-dir ./changer_r18_jilinone_workdir
```

3、模型推理

确保先前的训练过程中保存好了checkpoint，或使用本次比赛我们所提交的模型：best_score.pth。将该模型存放于open-cd-main文件夹中。

检查image_inference.py文件中的模型加载路径，数据加载路径，结果输出路径放置正确（若按照先前的步骤完成则无需更改）

```
# Load models into memory
inferencer = OpenCDInferencer(model='configs/changer/changer_ex_r18_256x256_40k_jilinone.py',
                               weights='best_score.pth',
                               classes=('0', '1', '2', '3', '4', '5', '6', '7', '8'), 
                               palette=[[255, 255, 0], [128, 128, 1], [130, 87, 2], [255, 0, 3],
                                         [0, 0, 4],[64,128,5],[64,128,6],[24,24,7],[100,200,8]])
# Inference
data_path = 'data/test/'
out_dir = 'OUTPUT_PATH'
```

请在终端输入以下命令：

```
python ./image_inference/image_inference.py
```

注：image_inference.py所消耗的时间分为两部分，一部分为模型推理时间，当命令行的进度条停止后这一部分完成，耗时很短。   
接下来，results和可视化结果会从显存保存至本地，这一部分的时间取决于测试电脑的I/O性能。

你会发现在项目目录中新生成了一个名叫OUTPUT_PATH的文件夹，这个文件夹中存放了对于复赛数据集的results和vis可视化结果。

```
|---OUTPUT_PATH
    |---results
        |---image_0.png
        |---image_1.png
        |---...
    |---vis
    |---pred(无需关注此文件夹)
```

## Citation
感谢changer团队所开发的opencd变化检测框架为本项目做出的贡献。
```bibtex
@ARTICLE{10129139,
  author={Fang, Sheng and Li, Kaiyu and Li, Zhe},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Changer: Feature Interaction is What You Need for Change Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2023.3277496}}
```
