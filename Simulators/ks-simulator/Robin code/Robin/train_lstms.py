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
from lorenz_map import LorenzMap
from lyapunov_exponents import stable_le, lyapunov_exponent_fulltraj_lstm
from lstms import Sequence, Sequence2, Sequence3

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
      (Correctly handling memory for lyapunov computation & graphs not done yet)
    """

    print(device)
    # set random seed to 0
    np.random.seed(0)

    us.dir_network(filename, None)
    logfile = filename+"/trainlog.txt"
    lyapfile = filename+"/lyapunovs.txt"

    model = seq[0]
    params = seq[1]
    seq = model(params[0], params[1], params[2], params[3], filename).double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(),  lr=0.1)

    #begin to train
    pos = [0, 1, 1.05]
    nb_steps = 54000
    delta_t = 5e-3

    str_settings = "Model : " + str(seq) + "\n\n"
    str_settings += "Loss : " + str(criterion) + "\n\n"
    str_settings += "Optimizer : " + str(optimizer) + "\n\n"
    str_settings += str(nb_steps)+" steps, delta t = "+str(delta_t) + ", init pos = " + str(pos)+"\n"
    us.save_settings(filename, str_settings)

    Lorenz = LorenzMap(delta_t=delta_t)
    data = Lorenz.full_traj(nb_steps, pos)
    data = data[:, np.newaxis, :]

    print(data.shape)
    len_train = data.shape[0]//1
    input  = torch.from_numpy(data[:len_train-1, :, :]).double().to(device)
    target = torch.from_numpy(data[1:len_train, :, :]).double().to(device)
    up.plot_traj(target, "Target", filename)


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
        us.save_trainlog(logfile, "Epoch " + str(i))
        torch.save(seq, './'+filename+'/'+filename+'_trained.pt')
        torch.save(seq.state_dict(), './'+filename+'/'+filename+'_training_state.pt')
        torch.save(optimizer.state_dict(), './'+filename+'/'+filename+'_optimizer.pt')

        if abs(loss_before - losses[-1]) < tol_change or i == max_epochs:
            if i == max_epochs:
                us.save_trainlog(logfile, 'Max epoch reached')
            else:
                us.save_trainlog(logfile, 'Converged')

            us.dir_jacobians(filename, "jacobians")
            jacobians = seq.jacobian_diff_tensor(input, seq(input)).detach().cpu().numpy()
            up.plot_singval1_lstm(data, jacobians, Lorenz, delta_t, nb_steps, 0, 5000, i, filename+"/jacobians")
            up.plot_losses(losses, filename, "all_losses")
            up.plot_traj(seq(input), "Final_pred", filename)

            with torch.no_grad():
                data_test = Lorenz.full_traj(100000, pos)[:, np.newaxis, :]
                input1 = torch.from_numpy(data_test[:-1, :, :]).double().to(device)
                target1 = torch.from_numpy(data_test[1:, :, :]).double().to(device)
                pred = seq(input1)
                loss = criterion(pred, target1)
                print('STEP TEST: ', i, 'test loss:', loss.item())

            jacobians = seq.jacobian_diff_tensor(input1, pred).detach().cpu().numpy()

            lyap_lstm = lyapunov_exponent_fulltraj_lstm(jacobians, delta_t)
            print("Lyapunov of NN model :", lyap_lstm)
            us.save_lyap(lyapfile, "Epoch "+str(i))
            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_lstm)+"\n")

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

            jacobians = seq.jacobian_diff_tensor(input1, pred).detach().cpu().numpy()

            lyap_lstm = lyapunov_exponent_fulltraj_lstm(jacobians, delta_t)
            print("Lyapunov of NN model :", lyap_lstm)
            us.save_lyap(lyapfile, "Epoch "+str(i))
            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_lstm)+"\n")

        if i == 1:
            data_test = Lorenz.full_traj(100000, pos)
            lyap_true_lorenz = stable_le(Lorenz.jacobian, data_test, delta_t=delta_t)
            print("Lyapunov of true Lorenz :", lyap_true_lorenz)
            us.save_lyap(lyapfile, "True lyapunov :"+str(lyap_true_lorenz)+"\n")

        print()

def train_loop(seq, filename, max_epochs, device, num_train=1):
    if os.path.exists(filename):
        shutil.rmtree(filename)
    for i in range(num_train):
        global losses
        losses = []
        trained = train(seq, "training_lstm", max_epochs, device)
        if trained:
            shutil.move("training_lstm", filename+"/training"+str(i))

def full_train():
    names = ["lstm0", "lstm0_2", "lstm0_3", "lstm1", "lstm2", "lstm3", "lstm4"]
    model0 = (Sequence, [50, 2, 3, 0])
    model0_2 = (Sequence2, [25, 1, 3, 0])
    model0_3 = (Sequence3, [50, 2, 3, 0])
    model1 = (Sequence, [10, 1, 3, 0])
    model2 = (Sequence, [100, 1, 3, 0])
    model3 = (Sequence, [10, 3, 3, 0])
    model4 = (Sequence, [30, 3, 3, 0])
    nets = [model0, model0_2, model0_3, model1, model2, model3, model4]

    for net, name in zip(nets, names):
        train_loop(net, name, 50, device)



if __name__ == '__main__':
    full_train()
