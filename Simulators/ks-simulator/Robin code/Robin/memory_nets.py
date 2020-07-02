import torch
import os
import copy
import shutil
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils_saving as us
import utils_plot as up
import utils_nets as un
from mlps import MLPs2
from lorenz_map import LorenzMap
from lyapunov_exponents import stable_le, lyapunov_exponent_fulltraj_NN

torch.set_default_tensor_type('torch.DoubleTensor')

device = un.device

class MLPmem(MLPs2):
    """
    NN with two hidden layers of 16 and 8
    ReLU activation
    """
    def __init__(self, mem_size):
        super().__init__()
        self.lin_mem = nn.Linear(mem_size, 1, bias=True)
        self.lin1 = nn.Linear(3, 16, bias=True)
        self.lin2 = nn.Linear(16, 3, bias=True)
        self.mem = mem_size

    def forward(self, xb):
        x = self.prep_data(xb)
        x = x.view(-1, self.mem)
        x = self.swish(self.lin_mem(x))
        x = x.view(-1, 3)
        x = self.swish(self.lin1(x))
        return self.lin2(x).view(-1, 1, 3)

    def prep_data(self, x):
        x = x.view(-1, 3)
        data = torch.zeros(x.size(0), x.size(1), self.mem)
        for i in range(data.size(0)):
            for j in range(x.size(1)):
                if i < self.mem:
                    data[i, j, :self.mem - i - 1] = x[0, j]
                    data[i, j, self.mem - i - 1:] = x[:i + 1, j]
                else:
                    data[i, j] = x[i - self.mem + 1:i + 1, j]
        return data



class MLPmem2(MLPs2):
    """
    NN with two hidden layers of 16 and 8
    ReLU activation
    """
    def __init__(self, mem_size):
        super().__init__()
        self.lin1 = nn.Linear(3 * mem_size, 64 * mem_size)
        self.lin2 = nn.Linear(64 * mem_size, 32 * mem_size)
        self.lin3 = nn.Linear(32 * mem_size, 3)
        self.mem = mem_size
        self.prev_data = None
        self.saved_data = None

    def forward(self, xb):
        if self.prev_data is None or not self.same_data(xb):
            self.prev_data = xb.clone()
            self.saved_data = self.prep_data(xb)

        x = self.saved_data
        x = x.view(-1, 3 * self.mem)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

    def same_data(self, x):
        if not x.size() == self.prev_data.size():
            return False
        else:
            return torch.all(torch.eq(x, self.prev_data))

    def prep_data(self, x):
        x = x.view(-1, 3)
        data = torch.zeros(x.size(0), x.size(1), self.mem)
        for i in range(data.size(0)):
            for j in range(x.size(1)):
                if i < self.mem:
                    data[i, j, :self.mem - i - 1] = x[0, j]
                    data[i, j, self.mem - i - 1:] = x[:i + 1, j]
                else:
                    data[i, j] = x[i - self.mem + 1:i + 1, j]
        return data

class MLPmem3(MLPmem2):
    """
    NN with two hidden layers of 16 and 8
    ReLU activation
    """
    def __init__(self, mem_size):
        super().__init__(mem_size)
        self.lin3 = nn.Linear(32 * self.mem, 32)
        self.lin4 = nn.Linear(32, 3)

    def forward(self, xb):
        if self.prev_data is None or not self.same_data(xb):
            self.prev_data = xb.clone()
            self.saved_data = self.prep_data(xb)

        x = self.saved_data
        x = x.view(-1, 3 * self.mem)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        x = self.swish(self.lin3(x))
        return self.lin4(x).view(-1, 1, 3)

class MLPmem4(MLPmem2):
    """
    NN with two hidden layers of 16 and 8
    ReLU activation
    """
    def __init__(self, mem_size):
        super().__init__(mem_size)
        self.lin1 = nn.Linear(3 * mem_size, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 3)

    def forward(self, xb):
        if self.prev_data is None or not self.same_data(xb):
            self.prev_data = xb.clone()
            self.saved_data = self.prep_data(xb)

        x = self.saved_data
        x = x.view(-1, 3 * self.mem)
        x = self.swish(self.lin1(x))
        x = self.swish(self.lin2(x))
        return self.lin3(x).view(-1, 1, 3)

num_pred = 0
losses = []
tol_change = 1e-7

def train(filename, max_epochs, mem_size, device):
    """
    Create a directory ./filename if it doesn't exist, where filename is a string
    In this directory will be saved :
      Some settings of the model in a file settings.txt
      States of the model and the optimizer
      The plot of the target data
    At the end of training :
      Compute lyapunov exponents at the end of each training
      Create a subdirectory jacobians where it puts graphs of the first singular values of jacobians
      The plot of the final prediction by the model, corresponding to the target data
    """
    print(device)
    # set random seed to 0
    np.random.seed(0)


    us.dir_network(filename, None)
    logfile = filename+"/trainlog.txt"
    lyapfile = filename+"/lyapunovs.txt"

    seq = MLPmem3(mem_size).double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.1)

    #begin to train
    pos = [0, 1, 1.05]
    nb_steps = 54000
    delta_t = 5e-3

    Lorenz = LorenzMap(delta_t=delta_t)
    data = Lorenz.full_traj(nb_steps, pos)
    data = data[:, np.newaxis, :]

    print(data.shape)
    len_train = data.shape[0]//1
    input  = torch.from_numpy(data[:len_train-1, :, :]).double().to(device)
    target = torch.from_numpy(data[1:len_train, :, :]).double().to(device)

    up.plot_traj(target, "Target", filename)

    str_settings = "Model : " + str(seq) + "\n\n"
    str_settings += "Loss : " + str(criterion) + "\n\n"
    str_settings += "Optimizer : " + str(optimizer) + "\n\n"
    str_settings += str(nb_steps)+" steps, delta t = "+str(delta_t) + ", init pos = " + str(pos)+"\n"
    str_settings += "Memory of "+str(mem_size)+" steps"

    us.save_settings(filename, str_settings)

    def closure():
        optimizer.zero_grad()
        out = seq(input)

        loss = criterion(out, target)
        print('loss:', loss.item())
        us.save_trainlog(logfile, 'loss:' + str(loss.item()))
        loss.backward()

        global losses
        losses.append(loss.item())

        return loss

    for i in range(1, max_epochs + 1):
        loss_before = 1e10 if i == 1 else losses[-1]
        optimizer.step(closure)
        print('EPOCH: ', i)
        us.save_trainlog(logfile, 'EPOCH: '+str(i))
        torch.save(seq, './'+filename+'/'+filename+'_trained.pt')
        torch.save(seq.state_dict(), './'+filename+'/'+filename+'_training_state.pt')
        torch.save(optimizer.state_dict(), './'+filename+'/'+filename+'_optimizer.pt')

        if abs(loss_before - losses[-1]) < tol_change or i == max_epochs:
            if i == max_epochs:
                us.save_trainlog(logfile, 'Max epoch reached')
            else:
                us.save_trainlog(logfile, 'Converged')

            us.dir_jacobians(filename, "jacobians")
            up.plot_singval1(seq, data, Lorenz, delta_t, nb_steps, 0, 5000, i, filename+"/jacobians")
            up.plot_losses(losses, filename, "all_losses")

            with torch.no_grad():
                data_test = Lorenz.full_traj(100000, pos)[:, np.newaxis, :]
                input1 = torch.from_numpy(data_test[:-1, :, :]).double().to(device)
                target1 = torch.from_numpy(data_test[1:, :, :]).double().to(device)
                pred = seq(input1)
                loss = criterion(pred, target1)
                print('STEP TEST: ', i, 'test loss:', loss.item())

                up.plot_traj(pred, "Final_pred", filename)

            if torch.cuda.is_available():
                preds = pred[:, 0, :].cpu().detach().numpy()
            else:
                preds = pred[:, 0, :].detach().numpy()

            lyap_nn = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t)
            print("Lyapunov of NN model :", lyap_nn)
            us.save_lyap(lyapfile, "Epoch "+str(i))
            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_nn)+"\n")

            return True

        if losses[-1] > 1e10 or np.isnan(losses[-1]):
            return False

        if i % 10 == 0:
            torch.save(seq, './'+filename+'/'+filename+'_trained_epoch'+str(i)+'.pt')
            with torch.no_grad():
                data_test = Lorenz.full_traj(100000, pos)[:, np.newaxis, :]
                input1 = torch.from_numpy(data_test[:-1, :, :]).double().to(device)
                target1 = torch.from_numpy(data_test[1:, :, :]).double().to(device)
                pred = seq(input1)
                loss = criterion(pred, target1)
                print('STEP TEST: ', i, 'test loss:', loss.item())

            if torch.cuda.is_available():
                preds = pred[:, 0, :].cpu().detach().numpy()
            else:
                preds = pred[:, 0, :].detach().numpy()

            lyap_nn = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t)
            print("Lyapunov of NN model :", lyap_nn)
            us.save_lyap(lyapfile, "Epoch "+str(i))
            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_nn)+"\n")

        if i == 1:
            data_test = Lorenz.full_traj(100000, pos)
            lyap_true_lorenz = stable_le(Lorenz.jacobian, data_test, delta_t=delta_t)
            print("Lyapunov of true Lorenz :", lyap_true_lorenz)
            us.save_lyap(lyapfile, "True lyapunov :"+str(lyap_true_lorenz)+"\n")

        print()


def train_loop(filename, max_epochs, mem_size, device, num_train=2):
    if os.path.exists(filename):
        shutil.rmtree(filename)
    for i in range(num_train):
        global losses
        losses = []
        trained = train("training_mem", max_epochs, mem_size, device)
        if trained:
            shutil.move("training_mem", filename+"/training"+str(i))

def full_train():

    mem = [1, 2, 4, 8, 16]

    for m in mem:
        name = "nn_mem"+str(m)
        train_loop(name, 50, m, device)


if __name__ == '__main__':
    full_train()
