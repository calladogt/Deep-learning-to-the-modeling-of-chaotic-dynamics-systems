import torch
import torch.nn as nn
import utils_nets as un
import torch.nn.functional as F

class Res_MLPs0(un.super_net):
    """
    NN with one hidden layer of 8
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__()
        self.delta_t = delta_t
        self.lin1 = nn.Linear(3, 3, bias=True)
        self.swish = un.Swish()

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        return xb + self.delta_t * x.view(-1, 1, 3)

class Res_MLPs1(Res_MLPs0):
    """
    NN with one hidden layer of 8
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        return xb + self.delta_t * self.lin2(x).view(-1, 1, 3)

class Res_MLPs2(Res_MLPs0):
    """
    NN with two hidden layers of 16 and 8
    Swish activation
    """
    def __init__(self, delta_t): 
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 16, bias=True)
        self.lin2 = nn.Linear(16, 8, bias=True)
        self.lin3 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return xb + self.delta_t * self.lin3(x).view(-1, 1, 3)

class Res_MLPs2_2(Res_MLPs2):
    """
    NN with two hidden layers of 32 and 16
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 32, bias=True)
        self.lin2 = nn.Linear(32, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

class Res_MLPs2_3(Res_MLPs2):
    """
    NN with two hidden layers of 64 and 32
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 32, bias=True)
        self.lin3 = nn.Linear(32, 3, bias=True)

class Res_MLPs2_4(Res_MLPs2):
    """
    NN with two hidden layers of 128 and 64
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 64, bias=True)
        self.lin3 = nn.Linear(64, 3, bias=True)

class Res_MLPs3(Res_MLPs0):
    """
    NN with three hidden layers of 8, 16, and 8
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 16, bias=True)
        self.lin3 = nn.Linear(16, 8, bias=True)
        self.lin4 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        x = self.swish(self.lin3(x))
        return xb + self.delta_t * self.lin4(x).view(-1, 1, 3)

class Res_MLPs3_2(Res_MLPs3):
    """
    NN with 3 hidden layers of 128, 256, and 128
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 256, bias=True)
        self.lin3 = nn.Linear(256, 128, bias=True)
        self.lin4 = nn.Linear(128, 3, bias=True)

class Res_MLPs4(Res_MLPs0):
    """
    NN with four hidden layers of 8, 16, 16, and 8
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 16, bias=True)
        self.lin3 = nn.Linear(16, 16, bias=True)
        self.lin4 = nn.Linear(16, 8, bias=True)
        self.lin5 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        x = self.swish(self.lin3(x))
        x = self.swish(self.lin4(x))
        return xb + self.delta_t * self.lin5(x).view(-1, 1, 3)

class Res_MLPs4_2(Res_MLPs4):
    """
    NN with 4 hidden layers of 256, 1024, 1024, and 256
    Swish activation
    """
    def __init__(self, delta_t):
        super().__init__(delta_t)
        self.lin1 = nn.Linear(3, 256, bias=True)
        self.lin2 = nn.Linear(256, 1024, bias=True)
        self.lin3 = nn.Linear(1024, 1024, bias=True)
        self.lin4 = nn.Linear(1024, 256, bias=True)
        self.lin5 = nn.Linear(256, 3, bias=True)
