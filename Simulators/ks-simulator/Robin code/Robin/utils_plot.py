import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
from numpy.linalg import eigvals
from numpy.linalg import svd

def plot_losses(losses, filename, title):
    """
    Plot losses during training
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses, label="loss training")
    plt.tick_params(labelsize=15)
    plt.legend(fontsize="x-large")
    plt.savefig("./"+filename+"/"+title+".png")
    plt.close()

def plot_traj(data, name, plot_dir):
    """
    Plot trajectory of the Lorenz system/prediction of the model
    in 3 dimensions with pyplot
    """
    if not torch.cuda.is_available():
        data = data.detach().numpy()
    else:
        data = data.cpu().detach().numpy()
    x, y, z = data[:, 0, 0], data[:, 0, 1], data[:, 0, 2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x, y, z, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(name)
    plt.savefig("./"+plot_dir+"/" + name)
    plt.close(fig)

def plot_traj_GAN(data, t, name, filename):
    """
    Plot trajectory of the Lorenz system/prediction of the model
    in 3 dimensions, along with position in the x coordinate
    """
    data = data.detach().cpu().numpy()
    t = t.detach().cpu().numpy()
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure(figsize=(15, 6), facecolor='white')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.plot(x, y, z, lw=0.5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("phase space")
    ax2.plot(t, x, lw=0.5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("X")
    ax2.set_title("time evol")
    fig.suptitle(name, fontsize=12)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig("./"+filename+"/plots/" + name)
    plt.close(fig)

def plot_traj_GAN_test(truth, data, t, name, filename):
    """
    Plot trajectory of the Lorenz system/prediction of the model
    in 3 dimensions, along with position in the x coordinate,
    for both truth and predictions together
    """
    criterion = torch.nn.MSELoss()
    loss = criterion(data, truth)

    truth = truth.detach().cpu().numpy()
    data = data.detach().cpu().numpy()
    t = t.detach().cpu().numpy()
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    xt, yt, zt = truth[:, 0], truth[:, 1], truth[:, 2]
    fig = plt.figure(figsize=(15, 12), facecolor='white')
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)

    ax1.plot(xt, yt, zt, lw=0.5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("phase space truth")
    ax2.plot(t, xt, lw=0.5)
    ax2.set_xlabel("time")
    ax2.set_ylabel("X")
    ax2.set_title("time evol truth")

    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)
    ax3.plot(x, y, z, lw=0.5)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("phase space\n loss = "+str(loss.item()))
    ax4.plot(t, x, lw=0.5)
    ax4.set_xlabel("time")
    ax4.set_ylabel("X")
    ax4.set_title("time evol")
    fig.suptitle(name, fontsize=12)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig("./"+filename+"/plots/" + name)
    plt.close(fig)

def plot_eigval1(seq, data, Lorenz, delta_t, nb_steps, beginning, nb_vals, epoch, jacob_dir):
    """
    Plot the evolution of the first eigenvalue on each data points,
    by plots containing each at most nb_vals data points,
    for the model and for the data (Lorenz system)
    """
    for begin in range(beginning, nb_steps, nb_vals):
        end = min(begin + nb_vals, len(data))
        eig_values = np.zeros((2, end - begin))
        for i in range(begin, end):
            jacob_autograd = seq.jacobian(data[i, 0])
            eig_values[0, i - begin] = np.absolute(eigvals(jacob_autograd)[0])

            jac_analyt = Lorenz.jacobian(data[i, 0])
            jac_analyt = expm(jac_analyt * delta_t)
            eig_values[1, i - begin] = np.absolute(eigvals(jac_analyt)[0])
        plt.figure(figsize=(15, 5))
        plt.plot(range(begin, end), eig_values[0], label="Autograd")
        plt.plot(range(begin, end), eig_values[1], label="Analytic")
        plt.tick_params(labelsize=20)
        plt.legend(fontsize="x-large")
        plt.savefig("./"+jacob_dir+"/epoch"+str(epoch)+"_points_"+str(begin)+"to"+str(begin+nb_vals)+".png")
        plt.close()

def plot_singval1(seq, data, Lorenz, delta_t, nb_steps, beginning, nb_vals, epoch, jacob_dir):
    """
    Plot the evolution of the first singular value on each data points,
    by plots containing each at most nb_vals data points,
    for the model and for the data (Lorenz system)
    """
    for begin in range(beginning, nb_steps, nb_vals):
        end = min(begin + nb_vals, len(data))
        svd_values = np.zeros((2, end - begin))
        for i in range(begin, end):
            jacob_autograd = seq.jacobian(data[i, 0])
            svd_values[0, i - begin] = svd(jacob_autograd)[1][0]

            jac_analyt = Lorenz.jacobian(data[i, 0])
            jac_analyt = expm(jac_analyt * delta_t)
            svd_values[1, i - begin] = svd(jac_analyt)[1][0]

        plt.figure(figsize=(15, 5))
        plt.plot(range(begin, end), svd_values[0], label="Autograd")
        plt.plot(range(begin, end), svd_values[1], label="Analytic")
        plt.tick_params(labelsize=20)
        plt.legend(fontsize="x-large")
        plt.savefig("./"+jacob_dir+"/epoch"+str(epoch)+"_points_"+str(begin)+"to"+str(end)+".png")
        plt.close()

def plot_singval_resnn(seq, data, Lorenz, delta_t, nb_steps, beginning, nb_vals, epoch, jacob_dir):
    """
    Plot the evolution of the first singular value on each data points,
    by plots containing each at most nb_vals data points,
    for the model and for the data (Lorenz system)
    """
    for begin in range(beginning, nb_steps, nb_vals):
        end = min(begin + nb_vals, len(data))
        svd_values = np.zeros((2, end - begin))
        for i in range(begin, end):
            jacob_autograd = seq.jacobian(data[i, 0]) * delta_t
            svd_values[0, i - begin] = svd(jacob_autograd)[1][0]

            jac_analyt = Lorenz.jacobian(data[i, 0])
            jac_analyt = expm(jac_analyt * delta_t)
            svd_values[1, i - begin] = svd(jac_analyt)[1][0]

        plt.figure(figsize=(15, 5))
        plt.plot(range(begin, end), svd_values[0], label="Autograd")
        plt.plot(range(begin, end), svd_values[1], label="Analytic")
        plt.tick_params(labelsize=20)
        plt.legend(fontsize="x-large")
        plt.savefig("./"+jacob_dir+"/epoch"+str(epoch)+"_points_"+str(begin)+"to"+str(end)+".png")
        plt.close()

def plot_intersingval(seq, data, Lorenz, delta_t, nb_steps, beginning, nb_vals, epoch, jacob_dir):
    """
    Plot the evolution of the first singular value on each data points,
    by plots containing each at most nb_vals data points,
    for the model and for the data (Lorenz system)
    """
    all_svd = []
    for begin in range(beginning, nb_steps, nb_vals):
        end = min(begin + nb_vals, len(data))
        svd_values = np.zeros((2, end - begin))
        for i in range(begin, end):
            jacob_autograd = seq.jacobian_l1_l2(data[i, 0])
            all_svd.append(svd(jacob_autograd)[1])
            svd_values[0, i - begin] = all_svd[-1][0]

            jac_analyt = Lorenz.jacobian(data[i, 0])
            jac_analyt = expm(jac_analyt * delta_t)
            svd_values[1, i - begin] = svd(jac_analyt)[1][0]

        plt.figure(figsize=(15, 5))
        plt.plot(range(begin, end), svd_values[0], label="Autograd")
        plt.plot(range(begin, end), svd_values[1], label="Analytic")
        plt.tick_params(labelsize=20)
        plt.legend(fontsize="x-large")
        plt.savefig("./"+jacob_dir+"/epoch"+str(epoch)+"_points_"+str(begin)+"to"+str(end)+".png")
        plt.close()
    filename = jacob_dir.split("/")[0]
    to_save = np.mean(all_svd, axis=0)
    np.savetxt(filename+"/svd_means.txt", to_save, delimiter=",")

def plot_singval1_lstm(data, jacobians, Lorenz, delta_t, nb_steps, beginning, nb_vals, epoch, jacob_dir):
    """
    Plot the evolution of the first singular value on each data points,
    by plots containing each at most nb_vals data points,
    for the model (jacobians need to be given in parameters)
    and for the data (Lorenz system)
    """
    for begin in range(beginning, nb_steps, nb_vals):
        end = min(begin + nb_vals, len(jacobians))
        svd_values = np.zeros((2, end - begin))
        for i in range(begin, end):
            jacob_autograd = jacobians[i]
            svd_values[0, i - begin] = svd(jacob_autograd)[1][0]

            jac_analyt = Lorenz.jacobian(data[i + 1, 0])
            jac_analyt = expm(jac_analyt * delta_t)
            svd_values[1, i - begin] = svd(jac_analyt)[1][0]

        plt.figure(figsize=(15, 5))
        plt.plot(range(begin, end), svd_values[0], label="Autograd")
        plt.plot(range(begin, end), svd_values[1], label="Analytic")
        plt.tick_params(labelsize=20)
        plt.legend(fontsize="x-large")
        plt.savefig("./"+jacob_dir+"/epoch"+str(epoch)+"_points_"+str(begin)+"to"+str(end)+".png")
        plt.close()

def plot_singval1_xpp(seq, data, Lorenz, delta_t, nb_steps, beginning, nb_vals, epoch, jacob_dir, out):
    """
    Plot the evolution of the first singular value on each data points,
    by plots containing each at most nb_vals data points,
    for the model and for the data (Lorenz system),
    as well as acceleration computed on data
    """
    xp = out[1:, 0, 0] - out[:-1, 0, 0]
    xpp = xp[1:] - xp[:-1]
    xpp = xpp.detach().numpy()
    for begin in range(beginning, nb_steps, nb_vals):
        end = min(begin + nb_vals, len(data))
        svd_values = np.zeros((2, end - begin))
        for i in range(begin, end):
            jacob_autograd = seq.jacobian(data[i, 0])
            svd_values[0, i - begin] = svd(jacob_autograd)[1][0]

            jac_analyt = Lorenz.jacobian(data[i, 0])
            jac_analyt = expm(jac_analyt * delta_t)
            svd_values[1, i - begin] = svd(jac_analyt)[1][0]

        plt.figure(figsize=(15, 5))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(range(begin, end), svd_values[0], label="Autograd")
        ax1.plot(range(begin, end), svd_values[1], label="Analytic")
        ax1.legend(fontsize=20)
        ax1.tick_params(labelsize=20)
        ax1.set_xticklabels([])
        ax2 = plt.subplot(2, 1, 2)
        xpp_begin = max(begin, 3)
        ax2.plot(range(xpp_begin, end), xpp[xpp_begin - 3: end - 3], label="Acceleration")
        ax2.tick_params(axis='both', labelsize=20)
        ax2.legend(fontsize=20)
        plt.savefig("./"+jacob_dir+"/epoch"+str(epoch)+"_points_"+str(begin)+"to"+str(end)+".png")
        plt.close()
