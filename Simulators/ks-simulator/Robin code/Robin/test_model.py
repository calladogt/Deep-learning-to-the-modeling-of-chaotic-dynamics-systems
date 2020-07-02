import torch
import argparse
import test_trajs
import torch.nn as nn
import numpy as np
import utils_saving as us
from lyapunov_exponents import lyapunov_exponent
from lyapunov_exponents import lyapunov_exponent_fulltraj_NN, lyapunov_exponent_fulltraj_lstm
from cnn import CNN
from lstm import Sequence, Sequence2, Sequence3
from basic_nn import MLP0, MLP1, MLP2, MLP3, MLPs, MLPs2
from nn_mgd import MLPs0, MLPs1, MLPs3, MLP_nnl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser('model testing')
parser.add_argument('--print_target', '-pt', type=eval, default=True, choices=[True, False])
parser.add_argument('--alone', '-a', type=eval, default=False, choices=[True, False])
parser.add_argument('--print_target_alone', '-pta', type=eval, default=True, choices=[True, False])
parser.add_argument('--nb_alone', '-nba', type=int, default=10000)
args = parser.parse_args()

torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device('cpu')
print(device)

def plot_traj(pos, data, path, lyap=None, loss=None):
    data = np.concatenate((pos[np.newaxis, :], data.reshape(-1, 3)), axis=0)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    str_pos = "Initial position : " + str(pos)
    str_lyap = "Value LE : " + str(lyap)
    title = str_pos + "\n" + str_lyap if not lyap is None else str_pos
    if not loss is None:
        title += "\n MSE Loss :" + str(loss)
    ax.set_title(title)
    plt.savefig(path)
    plt.close(fig)

def save_traj(data, namefile):
    np.savetxt(namefile, data)

if __name__ == '__main__':
    """
    Test trained models on different trajectories from different Lorenz systems
    Trajectory 0 is training data, 1 is a different origin, trajectories 2, 3, 4, 5 are Lorenz systems with different parameters but same origin
    """

    delta_t = 5e-3
    tested_trajs = [0, 1]

    criterion = torch.nn.MSELoss()
    TargetDone = {t:not args.print_target for t in tested_trajs}
    TestAlone = args.alone
    nbAlone = args.nb_alone
    TargetAloneDone = {t:not args.print_target_alone for t in tested_trajs}

    us.dir_test("test_models")

    ## Only work if models trained on Lorenz data with same delta t
    for model_name in ["nn0", "nn1", "mgd2", "cnn1"]:
        seq = test_trajs.get_model(model_name, device)

        for num_traj in [0, 1]:#[0, 1, 2, 3, 4, 5]: 
            Lorenz, pos = test_trajs.get_test_traj(num_traj, delta_t)

            subdir = "Different_systems/" if num_traj in [2, 3, 4, 5] else ""
            if not subdir == "":
                us.dir_network("test_models/"+subdir[:-1])
            print("Model ", model_name, " on trajectory", num_traj, "with correction enabled :", str(not TestAlone))

            if not TestAlone:

                # print("Chosen initial value :", pos)
                nb_steps = 100000

                data = Lorenz.full_traj(nb_steps, pos)

                loss = torch.zeros(1)

                with torch.no_grad():
                    input1  = torch.from_numpy(data[:-1, np.newaxis, :]).double().to(device)
                    target1 = torch.from_numpy(data[1:, np.newaxis, :]).double().to(device)
                    pred = seq(input1)
                    loss = criterion(pred, target1)

                target = data[1:, :]
                preds = pred[:, 0, :].detach().numpy()

                if not TargetDone[num_traj]:
                    TargetDone[num_traj] = True

                    input_lorenz = pos
                    lyap_lorenz = lyapunov_exponent(Lorenz.step, Lorenz.jacobian, input_lorenz, max_it=100000, delta_t=delta_t)
                    print("Lyapunov of true Lorenz :", lyap_lorenz)

                    path_target = "./test_models/"+subdir+"Target_traj"+str(num_traj)+"_dt_"+str(delta_t)+".png"
                    plot_traj(pos, target, path_target, lyap_lorenz)

                if seq.__class__ in [Sequence, Sequence2, Sequence3]:
                    jacob = seq.jacobian_diff_tensor(input1, pred)
                    lyap_model = lyapunov_exponent_fulltraj_lstm(jacob, delta_t)
                elif seq.__class__ is CNN:
                    lyap_model = "No LE for a CNN yet"
                else:
                    lyap_model = lyapunov_exponent_fulltraj_NN(seq.jacobian, preds, delta_t)
                print("Lyapunov of model :", lyap_model)

                path_pred = "./test_models/"+subdir+"Prediction_"+ model_name +"_traj"+str(num_traj)+".png"
                plot_traj(pos, preds, path_pred, lyap_model, loss.item())

            else:
                nb_normal = 1
                data = Lorenz.full_traj(nb_normal + nbAlone, pos)
                dataAfter = data[-nbAlone:]
                origin = data[-nbAlone - 1]
                loss = torch.zeros(1)

                with torch.no_grad():
                    input1  = torch.from_numpy(dataAfter[:-1, np.newaxis, :]).double().to(device)
                    target1 = torch.from_numpy(dataAfter[1:, np.newaxis, :]).double().to(device)
                    output1 = torch.zeros_like(target1)
                    output1[0] = seq(input1[0].unsqueeze(0))
                    for i in range(1, len(input1)):
                        if seq.__class__ in [Sequence, Sequence2, Sequence3]:
                            ## Correctly using memory still not implemented
                            output1[i] = seq(output1[i - 1].unsqueeze(0))
                        else:
                            output1[i] = seq(output1[i - 1].unsqueeze(0))
                    loss = criterion(output1, target1)

                us.dir_network("test_models/"+subdir+"After"+str(nb_normal))
                if not TargetAloneDone[num_traj]:
                    TargetAloneDone[num_traj] = True
                    path_target = "./test_models/"+subdir+"After"+str(nb_normal)+"/Target_traj"+str(num_traj)+".png"
                    plot_traj(origin, target1, path_target)
                path_pred = "./test_models/"+subdir+"After"+str(nb_normal)+"Prediction_"+ model_name +"_Traj"+str(num_traj)+".png"
                plot_traj(origin, output1, path_pred, loss=loss.item())
