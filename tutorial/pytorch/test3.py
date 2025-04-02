import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        x = x + y
        x = F.relu(x)
        return x

model = Model()
model.eval()

import allo
torch.compile(model, "allo")