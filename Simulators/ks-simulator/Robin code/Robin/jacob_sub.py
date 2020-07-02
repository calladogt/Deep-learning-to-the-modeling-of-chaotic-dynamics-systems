import torch
import numpy as np
import utils_saving as us
import utils_plot as up
import utils_nets as un
from lorenz_map import LorenzMap

torch.set_default_tensor_type('torch.DoubleTensor')

device = un.device

def eval_jacob(path_dir, filename, tf, subsampling, delta_t, device):
    """
    Scripts that are training for different parameters of subsampling
    don't render clearly singular values plots on jacobians.
    This one saves only the necessary number of the last singular values to always have
    a clear plot.
    """
    # print(device)
    # set random seed to 0
    np.random.seed(0)

    #begin to train
    pos = [0, 1, 1.05]
    nb_steps = 54000

    Lorenz = LorenzMap(delta_t=delta_t)
    data = Lorenz.full_traj(nb_steps * subsampling, pos)
    data = data[::subsampling, np.newaxis, :]

    dirs = [path_dir+filename+"/training0", path_dir+filename+"/training1"]
    for d in dirs:
        import os
        save_d = "latex_" + d[2:]
        if not os.path.exists(save_d):
            os.makedirs(save_d, exist_ok=True)
        if os.path.exists(d):
            seq = torch.load(d+'/'+tf).double().to(device)
            us.dir_jacobians(save_d, "jacobians")
            nb = min(int(5000 / (subsampling * delta_t / 0.005)), nb_steps)
            up.plot_singval1(seq, data, Lorenz, delta_t * subsampling, nb_steps, nb_steps - nb, nb, "_last", save_d+"/jacobians")
            print("Fig saved at:", "./"+(save_d+"/jacobians")+"/epoch_last_points_"+str(54000 - nb)+"to"+str(nb_steps)+".png")

def eval_jacob_cnn(path_dir, filename, tf, subsampling, delta_t, device):
    """
    Scripts that are training for different parameters of subsampling
    don't render clearly singular values plots on jacobians.
    This one saves only the necessary number last singular values to always have
    a clear plot.
    """
    # print(device)
    # set random seed to 0
    np.random.seed(0)

    #begin to train
    pos = [0, 1, 1.05]
    nb_steps = 54000

    Lorenz = LorenzMap(delta_t=delta_t)
    data = Lorenz.full_traj(nb_steps * subsampling, pos)
    data = data[::subsampling, np.newaxis, :]
    input = torch.from_numpy(data[:-1, :, :]).double().to(device)

    dirs = [path_dir+filename+"/training0", path_dir+filename+"/training1"]
    for d in dirs:
        import os
        save_d = "latex_" + d[2:]
        if os.path.exists(d):
            if not os.path.exists(save_d):
                os.makedirs(save_d, exist_ok=True)
            seq = torch.load(d+'/'+tf).double().to(device)
            us.dir_jacobians(save_d, "jacobians")
            nb = min(int(5000 / (subsampling * delta_t / 0.005)), nb_steps)
            jacobians = seq.jacobian_diff_tensor(input, seq(input)).detach().numpy()
            up.plot_singval1_lstm(data, jacobians, Lorenz, delta_t * subsampling, nb_steps, nb_steps - nb, nb, "_last", save_d+"/jacobians")
            print("Fig saved at:", "./"+(save_d+"/jacobians")+"/epoch_last_points_"+str(54000 - nb)+"to"+str(nb_steps)+".png")


def jacob_loop(path_dir, filename, trained_file, subsampling, delta_t, device):
    ## First line is for NNs/MLPs, second is for CNNs/LSTMs
    # eval_jacob(path_dir, filename, trained_file, subsampling, delta_t, device)
    eval_jacob_cnn(path_dir, filename, trained_file, subsampling, delta_t, device)


def full_jacob():
    delta_t = [1e-2, 5e-3, 1e-3, 1e-4]
    subsampling = [1, 3, 5, 10, 50, 100]

    ## put the correct dir / filenames, ...
    ## in the following lines
    path_dir = './cnn_sub_test/'
    trained_file = 'training_cnn_sub_trained.pt'

    for dt in delta_t:
        for ssp in subsampling:
            name = "cnn_sub"+str(ssp)+"_dt"+str(dt)
            jacob_loop(path_dir, name, trained_file, ssp, dt, device)


if __name__ == '__main__':
    full_jacob()
