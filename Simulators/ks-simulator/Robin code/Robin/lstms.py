import torch
import torch.nn as nn
import numpy as np
import utils_nets as un

device = un.device

class Sequence(un.super_net):
    def __init__(self, hidden, layer, features, dropout, filename):
        super().__init__()
        self.hidden = hidden
        self.layer = layer
        self.features = features
        self.filename = filename

        self.lstm1  = nn.LSTM(self.features, self.hidden, self.layer, dropout=dropout)
        self.dense = nn.Linear(self.hidden, self.features)

    def forward(self, x, test=False):
        h_t = torch.zeros(self.layer, x.size(1), self.hidden, dtype=torch.double).to(device)
        c_t = torch.zeros(self.layer, x.size(1), self.hidden, dtype=torch.double).to(device)
        if test:
            h_t = torch.from_numpy(np.load("./"+self.filename+"/state_h.npy")).to(device)
            c_t = torch.from_numpy(np.load("./"+self.filename+"/state_c.npy")).to(device)
        out, (h_t, c_t) = self.lstm1(x, (h_t, c_t))
        if not test:
            np.save("./"+self.filename+"/state_h.npy", h_t.detach().cpu().numpy())
            np.save("./"+self.filename+"/state_c.npy", c_t.detach().cpu().numpy())
        output = out.view(x.size(0)*x.size(1), self.hidden)
        output = self.dense(output)
        output = output.view(x.size(0), x.size(1), self.features)
        return output

    def step(self, x):
        tx = torch.from_numpy(x.reshape(1, 1, 3)).to(device)
        out = self.forward(tx, True)
        return out.detach()[0].cpu().numpy()

    def jacobian_diff_tensor(self, t, out):
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


class Sequence2(Sequence):
    def __init__(self, hidden, layer, features, dropout, filename):
        super().__init__(hidden, layer, features, dropout, filename)

        self.lstm_1 = nn.LSTM(self.features, self.hidden, self.layer, bidirectional=True, dropout=dropout)

        self.dense = nn.Sequential(nn.Linear(self.hidden * 2, self.hidden * 4),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden * 4, self.features))

    def forward(self, x, test=False):
        outputs = torch.zeros(0,0,0).to(device)
        h_t = torch.zeros(self.layer * 2, x.size(1), self.hidden, dtype=torch.double).to(device)
        c_t = torch.zeros(self.layer * 2, x.size(1), self.hidden, dtype=torch.double).to(device)
        if test:
            h_t = torch.from_numpy(np.load("./"+self.filename+"/state_h_"+self.filename+".npy")).to(device)
            c_t = torch.from_numpy(np.load("./"+self.filename+"/state_c_"+self.filename+".npy")).to(device)
        out, (h_t, c_t) = self.lstm1(x, (h_t, c_t))
        if not test:
            np.save("./"+self.filename+"/state_h_"+self.filename+".npy", h_t.detach().cpu().numpy())
            np.save("./"+self.filename+"/state_c_"+self.filename+".npy", c_t.detach().cpu().numpy())
        out = self.dense(out).view(x.size(0), x.size(1), self.features)
        return out


class Sequence3(Sequence):
    def __init__(self, hidden, layer, features, dropout, filename):
        super().__init__(hidden, layer, features, dropout, filename)

        self.lstm_1 = nn.LSTM(self.features, self.hidden, self.layer, dropout=dropout)

        self.dense = nn.Sequential(nn.Linear(self.hidden, self.hidden * 2),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden * 2, self.features))

    def forward(self, x, test=False):
        h_t = torch.zeros(self.layer, x.size(1), self.hidden, dtype=torch.double).to(device)
        c_t = torch.zeros(self.layer, x.size(1), self.hidden, dtype=torch.double).to(device)
        if test:
            h_t = torch.from_numpy(np.load("./"+self.filename+"/state_h_"+self.filename+".npy")).to(device)
            c_t = torch.from_numpy(np.load("./"+self.filename+"/state_c_"+self.filename+".npy")).to(device)
        out, (h_t, c_t) = self.lstm1(x, (h_t, c_t))
        if not test:
            np.save("./"+self.filename+"/state_h_"+self.filename+".npy", h_t.detach().cpu().numpy())
            np.save("./"+self.filename+"/state_c_"+self.filename+".npy", c_t.detach().cpu().numpy())
        out = self.dense(out).view(x.size(0), x.size(1), self.features)
        return out
