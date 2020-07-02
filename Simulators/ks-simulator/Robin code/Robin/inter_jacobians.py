import torch
import os
import shutil
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils_saving as us
import utils_plot as up
import utils_nets as un
import utils_mgd as umgd
from lorenz_map import LorenzMap
from lyapunov_exponents import stable_le, lyapunov_exponent_fulltraj_NN, lyapunov_exponent_fulltraj_interNN
from mlps import MLPeq2_0, MLPeq2, MLPeq2_2, MLPeq2_3, MLPeq2_4, MLPeq2_00, MLPeq2_01, MLPeq2_02


"""
File to train MLPs ith two layers and getting jacobians & lyapunov exponents
of the operation between the two intermediate layers
"""


torch.set_default_tensor_type('torch.DoubleTensor')

device = un.device

num_pred = 0
losses = []
tol_change = 1e-7

def train(seq, filename, max_epochs, device):
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

    seq = seq().double().to(device)
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

    J = umgd.jacobians_data(Lorenz, input, delta_t)

    up.plot_traj(target, "Target", filename)

    str_settings = "Model : " + str(seq) + "\n\n"
    str_settings += "Loss : " + str(criterion) + "\n\n"
    str_settings += "Optimizer : " + str(optimizer) + "\n\n"
    str_settings += str(nb_steps)+" steps, delta t = "+str(delta_t) + ", init pos = " + str(pos)

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
            up.plot_intersingval(seq, data, Lorenz, delta_t, nb_steps, 0, 5000, i, filename+"/jacobians")
            up.plot_losses(losses, filename, "all_losses")
            up.plot_traj(seq(input), "Final_pred", filename)

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

            lyap_nn_inter = lyapunov_exponent_fulltraj_interNN(seq.jacobian_l1_l2, preds, delta_t)
            print("Lyapunov intermediate layers of NN model :", lyap_nn_inter)
            us.save_lyap(lyapfile, "Lyapunov intermediate layers :"+str(lyap_nn_inter)+"\n")

            return True

        if losses[-1] > 1e10 or np.isnan(losses[-1]):
            return False

        if i % 20 == 0:
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

            lyap_nn_inter = lyapunov_exponent_fulltraj_interNN(seq.jacobian_l1_l2, preds, delta_t)
            print("Lyapunov intermediate layers of NN model :", lyap_nn_inter)
            us.save_lyap(lyapfile, "Lyapunov intermediate layers :"+str(lyap_nn_inter)+"\n")

        if i == 1:
            data_test = Lorenz.full_traj(100000, pos)
            lyap_true_lorenz = stable_le(Lorenz.jacobian, data_test, delta_t=delta_t)
            print("Lyapunov of true Lorenz :", lyap_true_lorenz)
            us.save_lyap(lyapfile, "True lyapunov :"+str(lyap_true_lorenz)+"\n")

        print()


def train_loop(seq, filename, max_epochs, device, num_train=3, max_failures=10):
    if os.path.exists(filename):
        shutil.rmtree(filename)
    i = 0
    fails = 0
    while i < num_train or fails == max_failures:
        global losses
        losses = []
        trained = train(seq, "training_intnn", max_epochs, device)
        if trained:
            shutil.move("training_intnn", filename+"/training"+str(i))
            i += 1
        else:
            fails += 1


def full_train():
    nets = [MLPeq2_00, MLPeq2_01, MLPeq2_02, MLPeq2_0, MLPeq2, MLPeq2_2, MLPeq2_3, MLPeq2_4]
    names = ["int_nns2_00", "int_nns2_01", "int_nns2_02", "int_nns2_0", "int_nns2", "int_nns2_2", "int_nns2_3", "int_nns2_4"]

    for net, name in zip(nets, names):
        train_loop(net, name, 100, device, 5)


if __name__ == '__main__':
    full_train()
