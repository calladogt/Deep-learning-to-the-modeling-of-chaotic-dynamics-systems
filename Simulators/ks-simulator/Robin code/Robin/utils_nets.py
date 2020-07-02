import torch
import torch.nn as nn
import numpy as np

# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     device = torch.device('cuda:0')
# else:
#     device = torch.device('cpu')

# device = torch.device('cuda:0')
device = torch.device('cpu')

class Swish(nn.Module):
    """
    Swish activation function
    => similar to ReLU without gradient error at 0
    """
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

class super_net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xb):
        return xb

    def step(self, x):
        tx = torch.from_numpy(x.reshape(1, 1, 3)).to(device)
        out = self.forward(tx)
        return out.detach()[0].cpu().numpy()

    def jacobian(self, x):
        J = []
        tx = torch.from_numpy(x.reshape(1, 1, 3)).to(device)
        tx.requires_grad = True
        F = self.forward(tx)
        size = len(tx[0, 0])
        weights = torch.eye(size).reshape(1, 1, size, size).to(device)
        for ind in range(size):
            J.append(torch.autograd.grad(F, tx, weights[:, :, ind], retain_graph=True)[0].cpu().numpy()[0, 0])
        J = np.array(J)
        return J

    def jacobian2(self, x):
        delta = 1e-3
        J = np.zeros((3, 3)).to(device)
        tx = torch.from_numpy(x.reshape(1, 1, 3)).to(device)
        y = self.forward(tx).detach().cpu().numpy()[0, 0]
        mat = torch.eye(3).to(device) * delta
        for k in range(len(J)):
            for l in range(len(J[k])):
                dy = self.forward(tx + mat[l])[0, 0].detach().cpu().numpy()
                J[k, l] = (dy[k] - y[k]) / delta
        return J

    def jacobian_diff_tensor(self, t, out):
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
