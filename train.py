import torch
import torch.optim as optim
import numpy as np

from model.vit import CONFIGS, VisionTransformer
from data_process.wch_cifar_dataloader import WCH_CIFAR10_dataloader
from data_process.wch_dataloader import WCH_dataloader
from data_process.data_utils import config_dataset
from utils import save_config, CL
from evaluate import evalModel

from tqdm import tqdm
import warnings
import random
import time
import sys
import os

warnings.filterwarnings("ignore")
# 设置多进程共享策略为文件系统
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config(start_time):
    config = {
        # "dataset": "mirflickr",
        "dataset": "cifar10-1",
        # "dataset": "coco",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_10",

        "info": "WCH",
        "bit_list": [16, 32, 64, 128],
        "backbone": "ViT-B_16",
        "pretrained_dir": "imagenet21k+imagenet2012_ViT-B_16-224.npz",
        "optimizer": {"type": optim.Adam, "lr": 1e-5},

        "epoch": 100,
        "test_map": 5,
        "batch_size": 16,
        "num_workers": 4,

        "logs_path": "logs",
        "resize_size": 256,
        "crop_size": 224,
        "alpha": 0.1,
    }
    config = config_dataset(config)
    config["logs_path"] = os.path.join(config["logs_path"], config['info'], start_time)
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])

    if "cifar" in config["dataset"]:
        config["topK"] = 5000
    else:
        config["topK"] = 5000
    return config


def train(config, bit):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # log文件
    train_logfile = open(os.path.join(config['logs_path'], 'train_log.txt'), 'a')
    for key, value in config.items():
        train_logfile.write(f"{key}: {value}\n")
    train_logfile.write("\n\n")
    train_logfile.write(f"***** {config['info']} - {config['backbone']} - {bit}bit *****\n\n")

    # 数据加载
    if "cifar" in config['dataset']:
        train_loader, test_loader, database_loader, num_train, num_test, num_database = WCH_CIFAR10_dataloader(config)
    else:
        train_loader, test_loader, database_loader, num_train, num_test, num_database = WCH_dataloader(config)

    Best_mAP = 0
    vit_config = CONFIGS[config['backbone']]
    vit_config.pretrained_dir = config['pretrained_dir']

    # vit参数，并加载预训练权重
    model = VisionTransformer(vit_config, 224, num_classes=config['n_class'], zero_head=True, hash_bit=bit).to(device)
    model.load_from(np.load(vit_config.pretrained_dir))

    # 优化器：Adam, 初始lr: 1e-5
    optimizer = config["optimizer"]["type"]([{"params": model.parameters(), "lr": config["optimizer"]["lr"]}])
    # 余弦退火调度器,根据余弦函数的形状调整学习率，在训练过程中逐渐降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epoch"])
    # 损失函数
    criterion = CL(config, bit)

    start_times = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print(f"\033[31m model:{config['info']} start_time:[{start_times}] bit:{bit}, dataset:{config['dataset']} \033[0m")
    print("\033[31m-----------------Training begins-----------------\033[0m")

    for epoch in range(config["epoch"]):
        model.train()
        epoch_loss = 0
        data_loader = tqdm(train_loader, file=sys.stdout)

        for step, (image1, image2) in enumerate(data_loader):
            image1, image2 = image1.to(device), image2.to(device)
            optimizer.zero_grad()

            h1, h2, weight = model.train_forward(image1, image2)

            loss = criterion(h1, h2, weight)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            data_loader.desc = f"[train epoch {epoch}] loss: {epoch_loss / (step + 1):.5f}"

        epoch_loss = epoch_loss / len(train_loader)
        scheduler.step()

        train_logfile.write(f"Epoch {epoch + 1}: Train Loss = {epoch_loss}\n")

        if (epoch + 1) % config["test_map"] == 0:
            model.eval()
            with torch.no_grad():
                Best_mAP = evalModel(test_loader, database_loader, model, Best_mAP, bit, config, epoch, train_logfile)

    print("\033[31m-----------------Training ended-----------------\033[0m")
    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print(f"\033[31m model:{config['info']} end_time:[{end_time}] bit:{bit}, dataset:{config['dataset']} \033[0m")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    # setup_seed(1234)  # 设置随机种子，确保结果可复现

    config = get_config(start_time)
    save_config(config, config["logs_path"])

    for bit in config["bit_list"]:
        train(config, bit)
        config = get_config(start_time)
