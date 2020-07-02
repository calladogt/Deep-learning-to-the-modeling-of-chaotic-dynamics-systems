import torch
import os
import shutil
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lorenz_map import LorenzMap
import utils_saving as us
import utils_plot as up
import utils_mgd as umgd
import utils_nets as un
from mgd_nets import MLPs2
from lyapunov_exponents import stable_le, lyapunov_exponent_fulltraj_NN

torch.set_default_tensor_type('torch.DoubleTensor')

device = un.device

num_pred = 0
losses = []
tol_change = 1e-10

def train(filename, max_epochs, subsampling, delta_t, device):
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

    seq = MLPs2().double().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.1)
    optimizer.param_groups[0]['tolerance_grad'] = 1e-10
    optimizer.param_groups[0]['tolerance_change'] = 1e-12

    #begin to train
    pos = [0, 1, 1.05]
    nb_steps = 54000

    Lorenz = LorenzMap(delta_t=delta_t)
    data = Lorenz.full_traj(nb_steps * subsampling, pos)
    data = data[::subsampling, np.newaxis, :]

    print(data.shape)
    len_train = data.shape[0]//1
    input  = torch.from_numpy(data[:len_train-1, :, :]).double().to(device)
    target = torch.from_numpy(data[1:len_train, :, :]).double().to(device)

    up.plot_traj(target, "Target", filename)

    str_settings = "Model : " + str(seq) + "\n\n"
    str_settings += "Loss : " + str(criterion) + "\n\n"
    str_settings += "Optimizer : " + str(optimizer) + "\n\n"
    str_settings += str(nb_steps)+" steps, delta t = "+str(delta_t) + ", init pos = " + str(pos)

    us.save_settings(filename, str_settings)

    J = umgd.jacobians_data(Lorenz, input, delta_t)

    def closure_shared_2():

        optimizer.zero_grad()

        input.requires_grad_(True)
        out = seq(input)

        loss1 = criterion(out, target)
        loss1.backward(retain_graph=True)
        print("Loss traj =", loss1.item())
        us.save_trainlog(logfile, "Loss traj = " + str(loss1.item()))

        grad1 = []
        for p in seq.parameters():
            grad1.append(p.grad.data.clone())

        optimizer.zero_grad()

        J_model = seq.jacobian_diff_tensor(input, out)

        loss2 = criterion(J_model, J)
        loss2.backward(retain_graph=True)
        print("Loss jacobians =", loss2.item())
        us.save_trainlog(logfile, "Loss jacobians = " + str(loss2.item()))

        grad2 = []
        for p in seq.parameters():
            grad2.append(p.grad.data.clone())

        optimizer.zero_grad()

        loss_shared = 0

        alpha = umgd.get_alpha2(grad1, grad2)

        loss_shared = alpha[0] * loss1 + alpha[1] * loss2
        print("Shared loss :", loss_shared.item(), " with alpha =", alpha)
        us.save_trainlog(logfile, "Shared loss : " + str(loss_shared.item()) + " with alpha = " + str(alpha))
        loss_shared.backward()
        print()
        us.save_trainlog(logfile, "")

        global losses
        losses.append([loss1.item(), loss2.item(), loss_shared.item()])

        return loss_shared

    for i in range(1, max_epochs + 1):
        loss_before = 1e10 if i == 1 else losses[-1][2]
        optimizer.step(closure_shared_2)
        print('EPOCH: ', i)
        us.save_trainlog(logfile, 'EPOCH: '+str(i))
        torch.save(seq, './'+filename+'/'+filename+'_trained.pt')
        torch.save(seq.state_dict(), './'+filename+'/'+filename+'_training_state.pt')
        torch.save(optimizer.state_dict(), './'+filename+'/'+filename+'_optimizer.pt')

        if abs(loss_before - losses[-1][2]) < tol_change or i == max_epochs:
            if i == max_epochs:
                us.save_trainlog(logfile, 'Max epoch reached')
            else:
                us.save_trainlog(logfile, 'Converged')

            us.dir_jacobians(filename, "jacobians")
            up.plot_singval1(seq, data, Lorenz, delta_t * subsampling, nb_steps, 0, 5000, i, filename+"/jacobians")
            l = np.array(losses)
            up.plot_losses(l[:, 0], filename, "loss_traj")
            up.plot_losses(l[:, 1], filename, "loss_jacob")
            up.plot_losses(l[:, 2], filename, "loss_shared")

            with torch.no_grad():
                data_test = Lorenz.full_traj(100000 * subsampling, pos)[::subsampling, np.newaxis, :]
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

            lyap_nn = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t * subsampling)
            print("Lyapunov of NN model :", lyap_nn)
            us.save_lyap(lyapfile, "Epoch "+str(i))
            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_nn)+"\n")

            return True

        if losses[-1][2] > 1e10 or np.isnan(losses[-1][2]):
            return False

        if i % 20 == 0:
            torch.save(seq, './'+filename+'/'+filename+'_trained_epoch'+str(i)+'.pt')
            with torch.no_grad():
                data_test = Lorenz.full_traj(100000 * subsampling, pos)[::subsampling, np.newaxis, :]
                input1 = torch.from_numpy(data_test[:-1, :, :]).double().to(device)
                target1 = torch.from_numpy(data_test[1:, :, :]).double().to(device)
                pred = seq(input1)
                loss = criterion(pred, target1)
                print('STEP TEST: ', i, 'test loss:', loss.item())

            if torch.cuda.is_available():
                preds = pred[:, 0, :].cpu().detach().numpy()
            else:
                preds = pred[:, 0, :].detach().numpy()

            lyap_nn = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t * subsampling)
            print("Lyapunov of NN model :", lyap_nn)
            us.save_lyap(lyapfile, "Epoch "+str(i))
            us.save_lyap(lyapfile, "Lyapunov of NN model :"+str(lyap_nn)+"\n")

        if i == 1:
            data_test = Lorenz.full_traj(100000 * subsampling, pos)[::subsampling, :]
            lyap_true_lorenz = stable_le(Lorenz.jacobian, data_test, delta_t=delta_t * subsampling)
            print("Lyapunov of true Lorenz :", lyap_true_lorenz)
            us.save_lyap(lyapfile, "True lyapunov :"+str(lyap_true_lorenz)+"\n")

        print()


def train_loop(filename, max_epochs, subsampling, delta_t, device, num_train=2):
    if os.path.exists(filename):
        shutil.rmtree(filename)
    for i in range(num_train):
        global losses
        losses = []
        trained = train("training_mgd_sub", max_epochs, subsampling, delta_t, device)
        if trained:
            shutil.move("training_mgd_sub", filename+"/training"+str(i))

def full_train():
    delta_t = [1e-2, 5e-3, 1e-3, 1e-4]
    subsampling = [1, 3, 5, 10, 50, 100]

    for dt in delta_t:
        for ssp in subsampling:
            name = "mgd_sub"+str(ssp)+"_dt"+str(dt)
            train_loop(name, 50, ssp, dt, device)



if __name__ == '__main__':
    full_train()
