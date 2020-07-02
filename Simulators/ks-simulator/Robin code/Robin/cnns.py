import torch
import torch.nn as nn
import utils_nets as un

device = un.device

"""
Deprecated file
Better to use cnns from causal_cnns.py
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.swish = un.Swish()

        self.c1 = nn.Conv1d(3, 16, 5, padding=2)
        self.c2 = nn.Conv1d(16, 16, 5, padding=2)
        self.out = nn.Linear(16, 3)

    def forward(self, inputs):
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d layers
        c = self.swish(self.c1(inputs))
        c = self.swish(self.c2(c))

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for output
        p = c.transpose(1, 2).transpose(0, 1)
        output = self.out(p)
        return output

    def jacobian_diff_tensorv1(self, t, out):
        """
        Compute jacobians on the whole trajectory
        delta_t is added to each data points so method not
        exactly correct
        Should only add on the evaluated points, but much more
        computationally expensive
        """
        delta = 1e-3
        J = torch.zeros(t.size()[0], 3, 3).to(device)

        mat = torch.zeros(3, t.size()[0], 1, 3).to(device)
        mat[0].add_(torch.tensor([1, 0, 0], dtype=torch.double, device=device))
        mat[1].add_(torch.tensor([0, 1, 0], dtype=torch.double, device=device))
        mat[2].add_(torch.tensor([0, 0, 1], dtype=torch.double, device=device))
        mat.mul_(delta)

        dout = torch.zeros(3, t.size()[0], 1, 3).to(device)
        for i in range(3):
            dout[i] = self.forward(t + mat[i])

        for k in range(3):
            for l in range(3):
                J[:, k, l] = (dout[l, :, 0, k] - out[:, 0, k]) / delta

        return J

class CNN_2(CNN):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 32, 5, padding=2)
        self.c2 = nn.Conv1d(32, 16, 5, padding=2)
        self.out = nn.Linear(16, 3)

class CNN_3(CNN):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 64, 5, padding=2)
        self.c2 = nn.Conv1d(64, 32, 5, padding=2)
        self.out = nn.Linear(32, 3)

class CNN_4(CNN):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 128, 5, padding=2)
        self.c2 = nn.Conv1d(128, 64, 5, padding=2)
        self.out = nn.Linear(64, 3)

class CNN_5(CNN):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 256, 5, padding=2)
        self.c2 = nn.Conv1d(256, 128, 5, padding=2)
        self.out = nn.Linear(128, 3)

class CNN_6(CNN):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 1024, 5, padding=2)
        self.c2 = nn.Conv1d(1024, 512, 5, padding=2)
        self.out = nn.Linear(512, 3)

class CNN4(CNN):
    def __init__(self):
        super().__init__()

        self.swish = un.Swish()

        self.c1 = nn.Conv1d(3, 128, 5, padding=2)
        self.c2 = nn.Conv1d(128, 128, 5, padding=2)
        self.c3 = nn.Conv1d(128, 64, 5, padding=2)
        self.out = nn.Linear(64, 3)

    def forward(self, inputs):
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d layers
        c = self.swish(self.c1(inputs))
        c = self.swish(self.c2(c))
        c = self.swish(self.c3(c))

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for output
        p = c.transpose(1, 2).transpose(0, 1)
        output = self.out(p)
        return output

class CNN4_2(CNN4):
    def __init__(self):
        super().__init__()

        self.swish = un.Swish()

        self.c1 = nn.Conv1d(3, 1024, 5, padding=2)
        self.c2 = nn.Conv1d(1024, 512, 5, padding=2)
        self.c3 = nn.Conv1d(512, 256, 5, padding=2)
        self.out = nn.Linear(256, 3)


class CNN5(CNN):
    def __init__(self):
        super().__init__()

        self.swish = un.Swish()

        self.c1 = nn.Conv1d(3, 128, 5, padding=2)
        self.c2 = nn.Conv1d(128, 128, 5, padding=2)
        self.c3 = nn.Conv1d(128, 64, 5, padding=2)
        self.c4 = nn.Conv1d(64, 64, 5, padding=2)
        self.out = nn.Linear(64, 3)

    def forward(self, inputs):
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d layers
        c = self.swish(self.c1(inputs))
        c = self.swish(self.c2(c))
        c = self.swish(self.c3(c))
        c = self.swish(self.c4(c))

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for output
        p = c.transpose(1, 2).transpose(0, 1)
        output = self.out(p)
        return output

class CNN5_2(CNN5):
    def __init__(self):
        super().__init__()

        self.swish = un.Swish()

        self.c1 = nn.Conv1d(3, 1024, 5, padding=2)
        self.c2 = nn.Conv1d(1024, 512, 5, padding=2)
        self.c3 = nn.Conv1d(512, 256, 5, padding=2)
        self.c4 = nn.Conv1d(256, 64, 5, padding=2)
        self.out = nn.Linear(64, 3)

class CNN2(CNN):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 16, 9, padding=4)
        self.c2 = nn.Conv1d(16, 16, 9, padding=4)

        self.out = nn.Linear(16, 3)


class CNN3(CNN):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv1d(3, 16, 15, padding=7)
        self.c2 = nn.Conv1d(16, 16, 15, padding=7)

        self.out = nn.Linear(16, 3)
