# 目录 

- [模型简介](#模型简介) 
- [模型效果](#模型效果) 
- [预训练权重](#预训练权重)
- [数据集](#数据集) 
- [如何运行](#如何运行)

# Weighted Contrastive Hashing

## 模型简介

WCH发表在[ACCV 2022](https://openaccess.thecvf.com/content/ACCV2022/html/Yu_Weighted_Contrative_Hashing_ACCV_2022_paper.html)。Weighted Contrastive Hashing 是一种用于图像检索和分类的先进算法。它通过细粒度的图像块交互来计算图像之间的相似性，并使用互注意力机制增强图像对的相似性，从而提高特征表示的质量。WCH 模型能够在对比学习中生成高质量的哈希编码，使得相似图像在哈希空间中更接近，非相似图像更远离。这种方法不仅提高了检索精度，还减少了误检样本的数量，尤其在高维度哈希编码下表现尤为出色。

<img src="/images/1.png" alt="image1" width="700" />

## 模型效果

**论文中的结果：**

<img src="\images\2.png" alt="2" width="700" />

<img src="\images\3.png" alt="3" width="700" />

<img src="\images\4.png" alt="4" width="700" />

**本人复现结果：**

<img src="\images\5.png" alt="6" width="700" />

<img src="\images\6.png" alt="6" width="700" />

## 预训练权重

模型需要`imagenet21k+imagenet2012_ViT-B_16-224`预训练权重。可以在[百度云盘（提取码：m003）](https://pan.baidu.com/s/1eGVoQUOgEi_RtBj15InevA#list/path=%2F)中下载。

## 数据集

**cifar10有三种不同的配置**

- config[“dataset”]=“cifar10”将使用1000个图像（每个类100个图像）作为查询集，5000个图像（每类500个图像）用作训练集，其余54000个图像用作数据库。
- config[“dataset”]=“cifar10-1”将使用1000个图像（每个类100个图像）作为查询集，其余59000个图像用作数据库，5000个图像（每类500个图像）从数据库中随机采样作为训练集。
- config[“dataset”]=“cifar10-2”将使用10000个图像（每个类1000个图像）作为查询集，50000个图像（每类5000个图像）用作训练集和数据库。

你可以在[这里](https://github.com/TreezzZ/DSDH_PyTorch)下载NUS-WIDE，它使用data/nus-wide/code.py进行划分，每个类随机选择100幅图像作为查询集（共2100幅图像）。剩余的图像被用作数据库集，我们从中每个类随机采样500个图像作为训练集（总共10500个图像）。

你可以在[这里](https://github.com/thuml/HashNet)下载ImageNet、NUS-WIDE-m和COCO数据集，或者使用[百度云盘](https://pan.baidu.com)（密码：hash）。NUS-WIDE中有269648个图像，其中195834个图像分为21个常见类别。NUS-WIDE-m有223496个图像。

你可以在[这里](https://www.liacs.nl/~mirflickr)下载mirflickr，然后使用data/mirflickr/code.py划分，随机选择1000个图像作为测试查询集和4000个图像作为训练集。

## 如何运行

1. ### 配置运行环境

   在自己所使用的环境中安装依赖库。

   ```python
   pip install requirements.txt
   ```

2. ### 数据集和预训练权重

   若使用CIFAR数据集，则无需自己下载，若使用其他数据集，则将下载的数据集放置在dataset文件夹下。

   将加载后的预训练权重放在文件夹内，并在train.py文件中修改调用路径。

3. ### 参数选择

   在train.py文件中修改运行参数。

   ```python
       config = {
           # "dataset": "mirflickr",
           "dataset": "cifar10-1",
           # "dataset": "coco",
           # "dataset": "nuswide_21",
           # "dataset": "nuswide_10",
   
           "info": "WCH",
           "bit_list": [16, 32, 64],
           "backbone": "ViT-B_16",
           "pretrained_dir": "imagenet21k+imagenet2012_ViT-B_16-224.npz",
           "optimizer": {"type": optim.Adam, "lr": 1e-5},
   
           "epoch": 50,
           "test_map": 10,
           "batch_size": 32,
           "num_workers": 4,
   
           "logs_path": "logs",
           "resize_size": 256,
           "crop_size": 224,
           "alpha": 0.1,
       }
   ```

4. ### 运行

   在终端输入命令：

   ```python
   python train.py
   ```

   

