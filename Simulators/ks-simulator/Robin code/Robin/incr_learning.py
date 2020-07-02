import torch
import time
import utils_nets as un
import utils_saving as us
import utils_plot as up
import utils_mgd as umgd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lorenz_map import LorenzMap
from lyapunov_exponents import lyapunov_exponent, lyapunov_exponent_fulltraj_NN


torch.set_default_tensor_type('torch.DoubleTensor')

device = un.device


class MLP(un.super_net):
    """
    MLP with two hidden layers of 64 and 16
    Swish activation between layers
    """
    def __init__(self, size_gen_alone):
        super().__init__()
        self.gen = size_gen_alone


        self.lin1 = nn.Linear(3, 64, bias=True)
        self.lin2 = nn.Linear(64, 16, bias=True)
        self.lin3 = nn.Linear(16, 3, bias=True)
        self.swish = un.Swish()

    def forward(self, xb):
        outputs = torch.zeros_like(xb)
        ## Mask to pick inputs that are used to compute one step ahead
        mask = (((torch.arange(outputs.size(0))) % self.gen) == 0).byte()
        ## First inputs used to predict
        inputs = xb[mask]
        ## Number of loops is equal to the horizon of prediction
        for i in range(self.gen):
            out = self.swish(self.lin1(inputs))
            out = self.swish(self.lin2(out))
            out = self.lin3(out)
            ## Mask to decide where to place the outputs
            mask = (((torch.arange(outputs.size(0)) + i) % self.gen) == 0).byte().view(-1, 1, 1).expand(-1, 1, 3)
            outputs.masked_scatter_(mask, out)
            ## Use predicted outputs as inputs for the next step
            inputs = out.clone()
        return outputs.view(-1, 1, 3)

class MLP_batch(MLP):
    """
    Same as MLP, but used when data are in batches instead of a long trajectory
    Batch size needs to be the same size as the horizon of prediction
    [Number of trajectories x size horizon x 3]
    """
    def __init__(self, size_gen_alone):
        super().__init__(size_gen_alone)

    def forward(self, xb):
        outputs = torch.zeros_like(xb)
        out = self.swish(self.lin1(xb[:, 0, :]))
        out = self.swish(self.lin2(out))
        outputs[:, 0, :] = self.lin3(out)
        for i in range(1, self.gen):
            out = self.swish(self.lin1(outputs[:, i - 1, :].clone()))
            out = self.swish(self.lin2(out))
            outputs[:, i, :] = self.lin3(out)
        return outputs

class LSTM(un.super_net):
    """
    Simple LSTM try for predicting multiple steps ahead
    """
    def __init__(self, size_gen_alone, hidden, layer, features, dropout=0.1):
        super().__init__()
        self.gen = size_gen_alone

        self.hidden = hidden
        self.layer = layer
        self.features = features

        self.lstm = nn.LSTM(self.features, self.hidden, self.layer, dropout=dropout)

        self.dense = nn.Sequential(nn.Linear(self.hidden, self.hidden * 2),
                                   un.Swish(),
                                   nn.Linear(self.hidden * 2, self.features))

    def forward(self, x):
        outputs = []
        h_t = torch.zeros(self.layer, x.size(1), self.hidden, dtype=torch.double).to(device)
        c_t = torch.zeros(self.layer, x.size(1), self.hidden, dtype=torch.double).to(device)

        out = x[0, :, :].unsqueeze(0)
        for i in range(x.size(0)):
            if i % self.gen == 0:
                out = x[i, :, :].unsqueeze(0)
            out, (h_t, c_t) = self.lstm(out, (h_t, c_t))
            out = self.dense(out)
            outputs += [out]
        outputs = torch.stack(outputs, 0)
        return outputs.view(-1, 1, 3)


if __name__ == '__main__':
    """
    Create a directory ./filename if it doesn't exist, where filename is a string
    In this directory will be saved :
      Some settings of the model in a file settings.txt
      States of the model and the optimizer
    In plot subdirectories will be saved :
      Target of predictions
      Plots of predicted trajectories
    At the end of training :
      Compute lyapunov exponents at the end of each training
      Create a subdirectory jacobians where it puts graphs of the first singular values of jacobians
    """

    # set random seed to 0
    np.random.seed(0)


    filename = "incr_learn_test"
    filename_saved = "incr_learn_test"
    step_train = 1
    us.dir_network(filename, "plots_step"+str(step_train))

    Restart = False
    steps_alone = 5
    # seq = LSTM(steps_alone, 5, 2, 3).double().to(device)
    seq = MLP(steps_alone).double().to(device)
    # seq = MLP_batch(steps_alone).double().to(device)
    criterion = nn.MSELoss()

    lr = 0.01
    optimizer = optim.LBFGS(seq.parameters(), lr=lr)
    if Restart or step_train > 1:
        seq.load_state_dict(torch.load('./'+filename+'/'+filename+'_step'+str(step_train-1)+'_training_state.pt'))
        optimizer.load_state_dict(torch.load('./'+filename+'/'+filename+'_step'+str(step_train-1)+'_optimizer.pt'))

    optimizer.param_groups[0]['lr'] = lr # Used to change lr to the wanted value after loading state of the optimizer
    # optimizer.param_groups[0]['tolerance_grad'] = 1e-10
    # optimizer.param_groups[0]['tolerance_change'] = 1e-12


    #begin to train
    pos = [0, 1, 1.05]
    nb_steps = 10000
    delta_t = 5e-3

    Lorenz = LorenzMap(delta_t=delta_t)
    data = Lorenz.full_traj(nb_steps, pos)
    data = data[:, np.newaxis, :]

    if seq.__class__ == MLP_batch:
        data = data.reshape(-1, steps_alone, 3)
    print(data.shape)
    len_train = data.shape[0]//1
    input  = torch.from_numpy(data[:len_train-1, :, :]).double().to(device)
    target = torch.from_numpy(data[1:len_train, :, :]).double().to(device)

    J = umgd.jacobians_data(Lorenz, input, delta_t)

    up.plot_traj(target.reshape(-1, 1, 3), "Target", filename+"/plots_step"+str(step_train))

    num_pred = 0

    max_epochs = 1

    str_settings = "Model : " + str(seq) + "\n\n"
    str_settings += "Loss : " + str(criterion) + "\n\n"
    str_settings += "Optimizer : " + str(optimizer) + "\n\n"
    str_settings += str(nb_steps) + " steps, delta t = " + str(delta_t) + ", init pos = " + str(pos) + "\n\n"
    str_settings += "Predicting " + str(steps_alone) + " steps alone\n\n"
    us.save_settings(filename, str_settings, step_train)

    def closure():
        ## Optimization function for single gradient descent
        optimizer.zero_grad()
        out = seq(input)

        ## Uncomment the followign if you want images of prediction every
        ## 20 training step
        # global num_pred
        # if num_pred % 20 == 0:
        #     up.plot_traj(out, "Step"+str(step_train)+"_Preds_" + str(num_pred), filename+"/plots_step"+str(step_train))
        # num_pred += 1

        loss = criterion(out, target)
        print('loss:', loss.item())
        us.save_trainlog(filename + "_step"+str(step_train), 'loss:' + str(loss.item()))
        loss.backward()

        return loss

    def closure_shared_2():
        ## Optimization function for multi-gradient descent

        optimizer.zero_grad()

        out = seq(input)

        ## Uncomment the followign if you want images of prediction every
        ## 20 training step
        # global num_pred
        # if num_pred % 20 == 0:
        #     up.plot_traj(out, "Preds_" + str(num_pred), filename+"/plots_step"+str(step_train))
        # num_pred += 1

        loss1 = criterion(out, target)
        loss1.backward(retain_graph=True)
        print("Loss traj =", loss1.item())
        us.save_trainlog(filename + "_step"+str(step_train), "Loss traj = " + str(loss1.item()))

        grad1 = []
        for p in seq.parameters():
            grad1.append(p.grad.data.clone())

        optimizer.zero_grad()

        J_model = seq.jacobian_diff_tensor(input, out)

        loss2 = criterion(J_model, J)
        loss2.backward(retain_graph=True)
        print("Loss jacobians =", loss2.item())
        us.save_trainlog(filename + "_step"+str(step_train), "Loss jacobians = " + str(loss2.item()))

        grad2 = []
        for p in seq.parameters():
            grad2.append(p.grad.data.clone())

        optimizer.zero_grad()

        loss_shared = 0

        alpha = umgd.get_alpha2(grad1, grad2)

        loss_shared = alpha[0] * loss1 + alpha[1] * loss2
        print("Shared loss :", loss_shared.item(), " with alpha =", alpha)
        us.save_trainlog(filename + "_step"+str(step_train), "Shared loss : "+str(loss_shared.item())+" with alpha = "+str(alpha))
        loss_shared.backward()
        print()
        us.save_trainlog(filename + "_step"+str(step_train), "")

        return loss_shared

    for i in range(1, max_epochs + 1):
        optimizer.step(closure)

        print('EPOCH: ', i)
        us.save_trainlog(filename + "_step"+str(step_train), "Epoch " + str(i))
        torch.save(seq.state_dict(), './'+filename+'/'+filename+'_step'+str(step_train)+'_training.pt')
        torch.save(optimizer.state_dict(), './'+filename+'/'+filename+'_step'+str(step_train)+'_optimizer.pt')

        if i == max_epochs:
            us.dir_jacobians(filename, "jacobians_step"+str(step_train))
            if data.shape[1] <= 1:
                up.plot_singval1(seq, data, Lorenz, delta_t, nb_steps, 0, 5000, i, filename+"/jacobians_step"+str(step_train))

                with torch.no_grad():
                    data_test = Lorenz.full_traj(100000, pos)[:, np.newaxis, :]
                    input1 = torch.from_numpy(data_test[:-1, :, :]).double().to(device)
                    target1 = torch.from_numpy(data_test[1:, :, :]).double().to(device)
                    pred = seq(input1)
                    loss = criterion(pred, target1)
                    print('STEP TEST: ', i, 'test loss:', loss.item())

                preds = pred[:, 0, :].detach().numpy()

                input_lorenz = input.detach()[0, 0].numpy()
                lyap_true_lorenz = lyapunov_exponent(Lorenz.step, Lorenz.jacobian, input_lorenz, max_it=100000, delta_t=delta_t)
                print("Lyapunov of true Lorenz :", lyap_true_lorenz)

                lyap_lstm = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t)
                print("Lyapunov of NN model :", lyap_lstm)
                us.save_trainlog(filename + "_step"+str(step_train), "Lyapunov of NN model :"+str(lyap_lstm))
            else:
                print("Not implemented yet for batches")
        print()
