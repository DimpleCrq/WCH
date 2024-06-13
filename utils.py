import json
import os
import torch.nn as nn
import torch


class MyEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, type):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


def save_config(config, save_path):
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, cls=MyEncoder, indent=4, separators=(', ', ': '))


class CL(nn.Module):
    def __init__(self, config, bit):
        super(CL, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bit = bit

    def forward(self, h1, h2, weighted):
        logit = torch.einsum('ik,jk->ij', h1, h2)
        logit = logit / self.bit / 0.3
        balance_logit = h1.sum(0) / h1.size(0)
        reg = self.mse(balance_logit, torch.zeros_like(balance_logit)) - self.mse(h1, torch.zeros_like(h1))
        loss = self.ce(logit, weighted) + reg
        return loss