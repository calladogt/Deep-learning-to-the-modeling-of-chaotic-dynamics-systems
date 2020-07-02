import torch
import torch.nn as nn
import utils_nets as un

class MLPs0(un.super_net):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 8, bias=True)
        self.lin3 = nn.Linear(8, 3, bias=True)
        self.swish = un.Swish()

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

class MLPs1(MLPs0):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 32, bias=True)
        self.lin2 = nn.Linear(32, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

class MLPs2(MLPs0):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

class MLPs3(MLPs0):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 32, bias=True)
        self.lin3 = nn.Linear(32, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

class MLPs4(MLPs0):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 64, bias=True)
        self.lin3 = nn.Linear(64, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

class MLP_nnl(un.super_net):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)
