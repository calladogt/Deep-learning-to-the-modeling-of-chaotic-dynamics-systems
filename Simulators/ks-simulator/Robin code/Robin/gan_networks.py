import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
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

class Generator(nn.Module):
    def __init__(self, size_gen):
        super(Generator, self).__init__()
        self.hidden = 50
        self.layer = 2
        self.features = 3
        self.size_gen = size_gen

        self.lstm1 = nn.LSTMCell(self.features, self.hidden)
        self.lstm2 = nn.LSTMCell(self.hidden, self.hidden)
        self.linear = nn.Linear(self.hidden, self.features)

    def forward(self, x):
        # x (seq, batch, features)
        outputs = []
        ht1 = torch.zeros(x.size(1), self.hidden).to(device)
        ct1 = torch.zeros(x.size(1), self.hidden).to(device)
        ht2 = torch.zeros(x.size(1), self.hidden).to(device)
        ct2 = torch.zeros(x.size(1), self.hidden).to(device)

        for i in range(x.size(0)):
            ht1, ct1 = self.lstm1(x[i, :, :], (ht1, ct1))
            ht2, ct2 = self.lstm2(ht1, (ht2, ct2))
            out = self.linear(ht2)
            outputs += [out]
        for i in range(self.size_gen-x.size(0)):
            ht1, ct1 = self.lstm1(out, (ht1, ct1))
            ht2, ct2 = self.lstm2(ht1, (ht2, ct2))
            out = self.linear(ht2)
            outputs += [out]
        outputs = torch.stack(outputs, 0)
        return outputs

    def pretrain(self, x):
        outputs = []
        ht1 = torch.zeros(x.size(1), self.hidden).to(device)
        ct1 = torch.zeros(x.size(1), self.hidden).to(device)
        ht2 = torch.zeros(x.size(1), self.hidden).to(device)
        ct2 = torch.zeros(x.size(1), self.hidden).to(device)
        for i in range(x.size(0)):
            ht1, ct1 = self.lstm1(x[i, :, :], (ht1, ct1))
            ht2, ct2 = self.lstm2(ht1, (ht2, ct2))
            out = self.linear(ht2)
            outputs += [out]
        outputs = torch.stack(outputs, 0)
        return outputs

class Generator2(Generator):
    """
    Neural network with two hidden layers of size 16 & 8
    with swish activation between layers
    Predict trajectory one step ahead for the input,
    Then for size size_gen - size input, generate trajectory alone given the
    previous output.
    """
    def __init__(self, size_gen):
        super().__init__(size_gen)

        self.lin1 = nn.Linear(3, 64, bias=True)
        self.swish = Swish()
        self.lin2 = nn.Linear(64, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)

    def forward(self, x):
        outputs = []
        for i in range(x.size(0)):
            out = self.swish(self.lin1(x[i, :, :]))
            out = self.swish(self.lin2(out))
            out = self.lin3(out)
            outputs += [out]
        for i in range(self.size_gen-x.size(0)):
            out = self.swish(self.lin1(out))
            out = self.swish(self.lin2(out))
            out = self.lin3(out)
            outputs += [out]
        return torch.stack(outputs)


    def pretrain(self, x):
        out = self.swish(self.lin1(x))
        out = self.swish(self.lin2(out))
        return self.lin3(out)

class Generator3(Generator2):
    """
    Neural network with two hidden layers of size 16 & 8
    with swish activation between layers
    Generating trajectory of size size_gen
    each n values (defined by spacing_truth), the network uses
    the true data instead of the previous input
    """
    def __init__(self, size_gen, spacing_truth):
        super().__init__(size_gen)

        self.spacing = spacing_truth

    def forward(self, x):
        outputs = []
        out = x[0, :, :]
        for i in range(self.size_gen):
            if i % self.spacing == 0:
                out = x[i, :, :]
            out = self.swish(self.lin1(out))
            out = self.swish(self.lin2(out))
            out = self.lin3(out)
            outputs += [out]
        return torch.stack(outputs)


class Generator4(nn.Module):
    def __init__(self, size_gen, input_size=3, hidden_size=20, output_size=3):
        super(Generator4, self).__init__()

        self.c1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.out = nn.Linear(hidden_size, output_size)
        self.swish = Swish()

    def forward(self, inputs):
        inputs = []

    def pretrain(self, x):
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = x.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d layers
        c = self.swish(self.c1(inputs))
        c = self.swish(self.c2(c))

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for output
        p = c.transpose(1, 2).transpose(0, 1)
        output = self.out(p)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.netD1 = nn.Sequential(
            nn.Conv2d(1, 20, (12, 3), stride=(4, 1)),
            Swish()
        )
        self.netD2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(20, 40, 9),
            Swish(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(40, 80, 5),
            Swish(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(12)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(80 * 12, 480),
            Swish(),
            nn.Dropout(),
            nn.Linear(480, 240),
            Swish(),
            nn.Linear(240, 1),
        )

    def forward(self, y):
        y = self.netD1(y)
        y = torch.squeeze(y, -1)
        y = self.netD2(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), 12 * 80)
        y = self.classifier(y)
        return y

class Discriminator2(Discriminator):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(80 * 12, 480),
            Swish(),
            nn.Dropout(),
            nn.Linear(480, 240),
            Swish(),
            nn.Linear(240, 1),
            nn.Sigmoid(),
        )

    def forward(self, y):
        y = self.netD1(y)
        y = torch.squeeze(y, -1)
        y = self.netD2(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), 12 * 80)
        y = self.classifier(y)
        return y
