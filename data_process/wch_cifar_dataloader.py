import numpy as np
import torch
import torchvision.datasets as dsets
from PIL import Image
from .data_utils import test_transform, train_transform


class Train_MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def WCH_CIFAR10_dataloader(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-1":
        test_size = 1000

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    # Dataset
    train_dataset = Train_MyCIFAR10(root=config["data_path"],
                              train=True,
                              transform=train_transform,
                              download=True)

    test_dataset = MyCIFAR10(root=config["data_path"],
                             train=False,
                             transform=test_transform)

    database_dataset = MyCIFAR10(root=config["data_path"],
                                 train=False,
                                 transform=test_transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass

    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))

    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index


    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = config['batch_size'],
                                               shuffle = True,
                                               num_workers = config['num_workers'])

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = config['batch_size'],
                                              shuffle = False,
                                              num_workers = config['num_workers'])

    database_loader = torch.utils.data.DataLoader(dataset = database_dataset,
                                                  batch_size = config['batch_size'],
                                                  shuffle = False,
                                                  num_workers = config['num_workers'])

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]