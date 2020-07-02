import torch
import torch.nn as nn
import utils_nets as un
import numpy as np
import torch.nn.functional as F

class MLP0(un.super_net):
    """
    NN with one hidden layer of 8
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = F.relu(self.lin1(x))
        return x.view(-1, 1, 3)

class MLP1(MLP0):
    """
    NN with one hidden layer of 8
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = F.relu(self.lin1(x))
        return self.lin2(x).view(-1, 1, 3)

class MLP2(MLP0):
    """
    NN with two hidden layers of 16 and 8
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 16, bias=True)
        self.lin2 = nn.Linear(16, 8, bias=True)
        self.lin3 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

class MLP2_2(MLP2):
    """
    NN with two hidden layers of 32 and 16
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 32, bias=True)
        self.lin2 = nn.Linear(32, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

class MLP2_3(MLP2):
    """
    NN with two hidden layers of 64 and 32
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 32, bias=True)
        self.lin3 = nn.Linear(32, 3, bias=True)

class MLP2_4(MLP2):
    """
    NN with two hidden layers of 128 and 64
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 64, bias=True)
        self.lin3 = nn.Linear(64, 3, bias=True)

class MLP3(MLP0):
    """
    NN with three hidden layers of 8, 16, and 8
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 16, bias=True)
        self.lin3 = nn.Linear(16, 8, bias=True)
        self.lin4 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x).view(-1, 1, 3)

class MLP3_2(MLP3):
    """
    NN with 3 hidden layers of 128, 256, and 128
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 256, bias=True)
        self.lin3 = nn.Linear(256, 128, bias=True)
        self.lin4 = nn.Linear(128, 3, bias=True)

class MLP4(MLP0):
    """
    NN with four hidden layers of 8, 16, 16, and 8
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 16, bias=True)
        self.lin3 = nn.Linear(16, 16, bias=True)
        self.lin4 = nn.Linear(16, 8, bias=True)
        self.lin5 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        return self.lin5(x).view(-1, 1, 3)

class MLP4_2(MLP4):
    """
    NN with 4 hidden layers of 256, 1024, 1024, and 256
    ReLU activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 256, bias=True)
        self.lin2 = nn.Linear(256, 1024, bias=True)
        self.lin3 = nn.Linear(1024, 1024, bias=True)
        self.lin4 = nn.Linear(1024, 256, bias=True)
        self.lin5 = nn.Linear(256, 3, bias=True)

class MLPs0(un.super_net):
    """
    NN with one hidden layer of 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3, bias=True)
        self.swish = un.Swish()

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        return x.view(-1, 1, 3)

class MLPs1(MLPs0):
    """
    NN with one hidden layer of 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        return self.lin2(x).view(-1, 1, 3)

class MLPs2(MLPs0):
    """
    NN with two hidden layers of 16 and 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 16, bias=True)
        self.lin2 = nn.Linear(16, 8, bias=True)
        self.lin3 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)


class MLPs2_2(MLPs2):
    """
    NN with two hidden layers of 32 and 16
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 32, bias=True)
        self.lin2 = nn.Linear(32, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

class MLPs2_3(MLPs2):
    """
    NN with two hidden layers of 64 and 32
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 32, bias=True)
        self.lin3 = nn.Linear(32, 3, bias=True)

class MLPs2_4(MLPs2):
    """
    NN with two hidden layers of 128 and 64
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 64, bias=True)
        self.lin3 = nn.Linear(64, 3, bias=True)

class MLPs3(MLPs0):
    """
    NN with three hidden layers of 8, 16, and 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 16, bias=True)
        self.lin3 = nn.Linear(16, 8, bias=True)
        self.lin4 = nn.Linear(8, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        x = self.swish(self.lin3(x))
        return self.lin4(x).view(-1, 1, 3)

class MLPs3_2(MLPs3):
    """
    NN with 3 hidden layers of 128, 256, and 128
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 256, bias=True)
        self.lin3 = nn.Linear(256, 128, bias=True)
        self.lin4 = nn.Linear(128, 3, bias=True)

class MLPs4(MLPs0):
    """
    NN with four hidden layers of 8, 16, 16, and 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
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
        return self.lin5(x).view(-1, 1, 3)

class MLPs4_2(MLPs4):
    """
    NN with 4 hidden layers of 256, 1024, 1024, and 256
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 256, bias=True)
        self.lin2 = nn.Linear(256, 1024, bias=True)
        self.lin3 = nn.Linear(1024, 1024, bias=True)
        self.lin4 = nn.Linear(1024, 256, bias=True)
        self.lin5 = nn.Linear(256, 3, bias=True)

class MLPsb0(MLPs0):
    """
    NN with one hidden layer of 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3, bias=False)

class MLPsb1(MLPs1):
    """
    NN with one hidden layer of 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=False)
        self.lin2 = nn.Linear(8, 3, bias=False)

class MLPsb2(MLPs2):
    """
    NN with two hidden layers of 16 and 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 16, bias=False)
        self.lin2 = nn.Linear(16, 8, bias=False)
        self.lin3 = nn.Linear(8, 3, bias=False)

class MLPsb2_2(MLPs2):
    """
    NN with two hidden layers of 32 and 16
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 32, bias=False)
        self.lin2 = nn.Linear(32, 16, bias=False)
        self.lin3 = nn.Linear(16, 3, bias=False)

class MLPsb2_3(MLPs2):
    """
    NN with two hidden layers of 64 and 32
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 64, bias=False)
        self.lin2 = nn.Linear(64, 32, bias=False)
        self.lin3 = nn.Linear(32, 3, bias=False)

class MLPsb2_4(MLPs2):
    """
    NN with two hidden layers of 128 and 64
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=False)
        self.lin2 = nn.Linear(128, 64, bias=False)
        self.lin3 = nn.Linear(64, 3, bias=False)

class MLPsb3(MLPs3):
    """
    NN with three hidden layers of 8, 16, and 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=False)
        self.lin2 = nn.Linear(8, 16, bias=False)
        self.lin3 = nn.Linear(16, 8, bias=False)
        self.lin4 = nn.Linear(8, 3, bias=False)

class MLPsb3_2(MLPs3):
    """
    NN with 3 hidden layers of 128, 256, and 128
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=False)
        self.lin2 = nn.Linear(128, 256, bias=False)
        self.lin3 = nn.Linear(256, 128, bias=False)
        self.lin4 = nn.Linear(128, 3, bias=False)

class MLPsb4(MLPs4):
    """
    NN with four hidden layers of 8, 16, 16, and 8
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=False)
        self.lin2 = nn.Linear(8, 16, bias=False)
        self.lin3 = nn.Linear(16, 16, bias=False)
        self.lin4 = nn.Linear(16, 8, bias=False)
        self.lin5 = nn.Linear(8, 3, bias=False)

class MLPsb4_2(MLPs4):
    """
    NN with 4 hidden layers of 256, 1024, 1024, and 256
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 256, bias=False)
        self.lin2 = nn.Linear(256, 1024, bias=False)
        self.lin3 = nn.Linear(1024, 1024, bias=False)
        self.lin4 = nn.Linear(1024, 256, bias=False)
        self.lin5 = nn.Linear(256, 3, bias=False)


class MLPeq2(MLPs0):
    """
    NN with two hidden layers of 16
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 16, bias=True)
        self.lin2 = nn.Linear(16, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

    def forward(self, xb):
        x = xb.view(-1, 3)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

    def jacobian_l1_l2(self, t):
        J = []
        tx = torch.from_numpy(t.reshape(1, 1, 3))
        inputs = self.swish(self.lin1(tx))
        out = self.lin2(inputs)

        weights = torch.eye(out.size(-1)).reshape(1, 1, out.size(-1), out.size(-1))
        for ind in range(out.size(-1)):
            J.append(torch.autograd.grad(out, inputs, weights[:, :, ind], retain_graph=True)[0].cpu().numpy()[0, 0])
        J = np.array(J)
        return J
class MLPeq2_00(MLPeq2):
    """
    NN with two hidden layers of 32
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 2, bias=True)
        self.lin2 = nn.Linear(2, 2, bias=True)
        self.lin3 = nn.Linear(2, 3, bias=True)

class MLPeq2_01(MLPeq2):
    """
    NN with two hidden layers of 32
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 4, bias=True)
        self.lin2 = nn.Linear(4, 4, bias=True)
        self.lin3 = nn.Linear(4, 3, bias=True)

class MLPeq2_02(MLPeq2):
    """
    NN with two hidden layers of 32
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 6, bias=True)
        self.lin2 = nn.Linear(6, 6, bias=True)
        self.lin3 = nn.Linear(6, 3, bias=True)

class MLPeq2_0(MLPeq2):
    """
    NN with two hidden layers of 32
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 8, bias=True)
        self.lin2 = nn.Linear(8, 8, bias=True)
        self.lin3 = nn.Linear(8, 3, bias=True)

class MLPeq2_2(MLPeq2):
    """
    NN with two hidden layers of 32
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 32, bias=True)
        self.lin2 = nn.Linear(32, 32, bias=True)
        self.lin3 = nn.Linear(32, 3, bias=True)

class MLPeq2_3(MLPeq2):
    """
    NN with two hidden layers of 64
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 64, bias=True)
        self.lin3 = nn.Linear(64, 3, bias=True)

class MLPeq2_4(MLPeq2):
    """
    NN with two hidden layers of 128
    Swish activation
    """
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128, bias=True)
        self.lin2 = nn.Linear(128, 128, bias=True)
        self.lin3 = nn.Linear(128, 3, bias=True)
