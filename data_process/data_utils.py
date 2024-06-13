from torchvision import transforms
import random
from PIL import ImageFilter



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = 5000
        config["n_class"] = 10
    elif config["dataset"] == "nuswide_21":
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_10":
        config["topK"] = 5000
        config["n_class"] = 10
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 24

    if "cifar" in config["dataset"]:
        config["data_path"] = "./dataset/cifar/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] == "nuswide_10":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] == "coco":
        config["data_path"] = "./dataset/coco/"
    if config["dataset"] == "mirflickr":
        config["data_path"] = "./dataset/flickr25k/mirflickr/"
    config["data_list"] = {
        "train_dataset": "./data/" + config["dataset"] + "/train.txt",
        "test_dataset": "./data/" + config["dataset"] + "/test.txt",
        "database_dataset": "./data/" + config["dataset"] + "/database.txt"}
    return config