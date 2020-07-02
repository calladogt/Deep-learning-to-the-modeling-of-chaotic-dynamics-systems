import matplotlib.pyplot as plt
import numpy as np
import os

prefix = "toKeep/"
nets = ["nn", "res_mlps", "cnn", "mgd", "lstm"]
subs = [1, 3, 5, 10, 50, 100]
dts = [1e-2, 5e-3, 1e-3, 1e-4]
lyap_file = "lyapunovs.txt"
loss_file = "trainlog.txt"

def plot_sub(cats, vals, labels, title, xlab="subsampling"):
    """
    Plot loss/distance to true lyapunovs depending on
    category, i.e. type of net or delta t
    """
    vals = np.array(vals)
    plt.figure(figsize=(15, 8))
    cpt = 0
    for val, label in zip(vals, labels):
        plt.loglog(cats, val, label=label)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize="large")
    plt.xlabel(xlab, fontsize="large")
    plt.savefig("./plots_res/subsampling/together/"+title+".png")
    plt.close()

    plt.figure(figsize=(20, 10))
    cpt = 0
    for val, label in zip(vals, labels):
        cpt += 1
        if cpt == 3:
            cpt += 1
        ax1 = plt.subplot(2, 2 if cpt <= 2 else 3, cpt)
        ax1.semilogx(cats, val, label=label)
        ax1.tick_params(labelsize=15)
        ax1.legend(fontsize="large")
        if cpt == 5:
            ax1.set_xlabel(xlab, fontsize="large")
    plt.savefig("./plots_res/subsampling/sideBside/"+title+".png")
    plt.close()


def best_dist_lyap(lyaps):
    true_le = np.array([0.905, 0, -14.57])
    lyaps = np.array(lyaps)
    # print(lyaps)
    if len(lyaps) == 1:
        return np.linalg.norm(true_le - lyaps[0])
    elif not None in lyaps:
        d1 = np.linalg.norm(true_le - lyaps[0])
        d2 = np.linalg.norm(true_le - lyaps[1])
        return min(d1, d2)
    elif lyaps[0] == None:
        if lyaps[1] == None:
            return None
        else:
            return np.linalg.norm(true_le - lyaps[1])
    else:
        return np.linalg.norm(true_le - lyaps[0])

def best_dist_lyap_2(lyaps, num_exposant=0):
    true_le = np.array([0.905, 0, -14.57])
    lyaps = np.array(lyaps)
    # print(lyaps)
    if len(lyaps) == 1:
        return np.linalg.norm(true_le[num_exposant] - lyaps[0][num_exposant])
    elif not None in lyaps:
        d1 = np.linalg.norm(true_le[num_exposant] - lyaps[0][num_exposant])
        d2 = np.linalg.norm(true_le[num_exposant] - lyaps[1][num_exposant])
        return min(d1, d2)
    elif lyaps[0] == None:
        if lyaps[1] == None:
            return None
        else:
            return np.linalg.norm(true_le[num_exposant] - lyaps[1][num_exposant])
    else:
        return np.linalg.norm(true_le[num_exposant] - lyaps[0][num_exposant])


def get_lyap(f):
    fl = f.readlines()
    last_lyap = fl[-2][:-1]
    last_lyap = last_lyap.split("[")[1].split("]")[0].split(" ")
    while "" in last_lyap:
        last_lyap.remove("")
    if "-inf" in last_lyap:
        return None
    else:
        return [float(le) for le in last_lyap]

def get_loss(f, net):
    fl = f.readlines()
    line = -6 if net == "mgd" else -3
    loss = fl[line][:-1]
    if net == "mgd":
        loss = loss.split("=")
        return float(loss[1][1:])
    else:
        loss = loss.split(":")
        return float(loss[1])

def plots_by_dt():
    for dt in dts:
        losses = []
        lyaps = []
        for d in nets:
            subdir = d + "_sub/"
            p1 = prefix + subdir
            losses_2 = []
            lyaps_2 = []
            for sub in subs:
                p2 = p1 + d + "_sub" + str(sub) + "_dt" + str(dt)
                if os.path.exists(p2):
                    trainings = os.listdir(p2 + "/")
                    l = []
                    le = []
                    for f in trainings:
                        p3 = p2 + "/" + f + "/"
                        # print(p3)
                        lyapf = open(p3 + lyap_file, "r")
                        le.append(get_lyap(lyapf))
                        lyapf.close()
                        lossf = open(p3 + loss_file, "r")
                        l.append(get_loss(lossf, d))
                        lossf.close()
                    if len(l) == 1:
                        losses_2.append(l[0])
                    else:
                        losses_2.append(min(l[0], l[1]))
                    lyaps_2.append(best_dist_lyap(le))
                else:
                    losses_2.append(None)
                    lyaps_2.append(None)
            losses.append(losses_2)
            lyaps.append(lyaps_2)
        plot_sub(subs, losses, ["loss "+d for d in nets], "by_dt/losses_dt" + str(dt))
        plot_sub(subs, lyaps, ["distance to true lyapunovs "+d for d in nets], "by_dt/lyapunovs_dt" + str(dt))

def plots_by_net():
    for d in nets:
        losses = []
        lyaps = []
        for dt in dts:
            subdir = d + "_sub/"
            p1 = prefix + subdir
            losses_2 = []
            lyaps_2 = []
            for sub in subs:
                p2 = p1 + d + "_sub" + str(sub) + "_dt" + str(dt)
                if os.path.exists(p2):
                    trainings = os.listdir(p2 + "/")
                    l = []
                    le = []
                    for f in trainings:
                        p3 = p2 + "/" + f + "/"
                        # print(p3)
                        lyapf = open(p3 + lyap_file, "r")
                        le.append(get_lyap(lyapf))
                        lyapf.close()
                        lossf = open(p3 + loss_file, "r")
                        l.append(get_loss(lossf, d))
                        lossf.close()
                    if len(l) == 1:
                        losses_2.append(l[0])
                    else:
                        losses_2.append(min(l[0], l[1]))
                    lyaps_2.append(best_dist_lyap(le))
                else:
                    losses_2.append(None)
                    lyaps_2.append(None)
            losses.append(losses_2)
            lyaps.append(lyaps_2)
        plot_sub(subs, losses, ["loss dt = "+str(dt) for dt in dts], "by_net/losses_" + d)
        plot_sub(subs, lyaps, ["distance to true lyapunovs dt = "+str(dt) for dt in dts], "by_net/lyapunovs_" + d)

def get_interLE(lyapfile):
    """
    Get first and last lyapunov exponents
    """
    fl = lyapfile.readlines()
    while "\n" in fl:
        fl.remove("\n")
    get_idx = 0
    for i, l in enumerate(fl):
        if l == "Epoch 100\n":
            get_idx = i
    fl = "".join(fl[get_idx + 2:])
    fl = fl.split("[")[1].split("]")[0].split("\n")
    fl = "".join(fl).split(" ")
    while "" in fl:
        fl.remove("")
    return [float(le) for le in fl]


def get_svds(svdfile):
    fl = svdfile.readlines()
    return [float(val) for val in fl]

def plots_svd_LE(svds, title):
    vals = np.array(svds)
    plt.figure(figsize=(15, 8))
    valdict = {len(val):val for val in vals}
    sort_dict = {key:valdict[key] for key in sorted(valdict)}

    plt.tick_params(labelsize=15)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(list(sort_dict.keys()), [v[0] for v in sort_dict.values()], label="First value")
    ax1.set_xlabel("Size of intermediate layers", fontsize="large")
    ax1.legend(fontsize="large")

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(list(sort_dict.keys()), [v[-1] for v in sort_dict.values()], label="Last value")
    ax2.set_xlabel("Size of intermediate layers", fontsize="large")
    ax2.legend(fontsize="large")

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(list(sort_dict.keys()), [np.mean(v) for v in sort_dict.values()], label="Mean value")
    ax3.set_xlabel("Size of intermediate layers", fontsize="large")
    ax3.legend(fontsize="large")

    ax4 = plt.subplot(2, 1, 2)
    ax4.plot(list(sort_dict.keys()), [v[0] for v in sort_dict.values()], label="First value")
    ax4.plot(list(sort_dict.keys()), [v[-1] for v in sort_dict.values()], label="Last value")
    ax4.plot(list(sort_dict.keys()), [np.mean(v) for v in sort_dict.values()], label="Mean value")
    ax4.set_xlabel("Size of intermediate layers", fontsize="large")
    ax4.legend(fontsize="large")

    plt.savefig("./plots_res/inter_layers/"+title+".png")
    plt.close()

def plots_interlayers():
    svd_file = "svd_means.txt"
    p1 = prefix + "inter_layers/"
    nets = os.listdir(p1)
    lyaps = []
    svds = []
    for net in nets:
        p2 = p1 + net
        trainings = os.listdir(p2 + "/")
        i = 0
        le = []
        sv = []
        for train in trainings:
            # Mean on trainings
            p3 = p2 + "/" + train + "/"
            lyapf = open(p3 + lyap_file, "r")
            le.append(get_interLE(lyapf))
            lyapf.close()
            svdf = open(p3 + svd_file, "r")
            sv.append(get_svds(svdf))
            svdf.close()
        lyaps.append(np.mean(le, axis=0))
        svds.append(np.mean(sv, axis=0))
    plots_svd_LE(lyaps, "Lyapunovs plot")
    plots_svd_LE(svds, "Singular values plot")

dts_resnn = [1e-1, 5e-2, 2.5e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]
version = ["resnet_mlp_" + vers for vers in ["sub1", "sub2", "sub3"]]

def plots_resnn_by_dt():
    for dt in dts_resnn:
        losses = []
        lyaps = []
        for d in version:
            p1 = prefix + d
            losses_2 = []
            lyaps_2 = []
            for sub in [1]:
                p2 = p1 + d + "_sub" + str(sub) + "_dt" + str(dt)
                if os.path.exists(p2):
                    trainings = os.listdir(p2 + "/")
                    l = []
                    le = []
                    for f in trainings:
                        p3 = p2 + "/" + f + "/"
                        # print(p3)
                        lyapf = open(p3 + lyap_file, "r")
                        le.append(get_lyap(lyapf))
                        lyapf.close()
                        lossf = open(p3 + loss_file, "r")
                        l.append(get_loss(lossf, d))
                        lossf.close()
                    if len(l) == 1:
                        losses_2.append(l[0])
                    else:
                        losses_2.append(min(l[0], l[1]))
                    lyaps_2.append(best_dist_lyap(le))
                else:
                    losses_2.append(None)
                    lyaps_2.append(None)
            losses.append(losses_2)
            lyaps.append(lyaps_2)
        plot_sub(dts_resnn, losses, ["loss "+d for d in version], "by_dt/losses_dt" + str(dt), "Delta t")
        plot_sub(dts_resnn, lyaps, ["distance to true lyapunovs "+d for d in version], "by_dt/lyapunovs_dt" + str(dt), "Delta t")

if __name__ == "__main__":
    # plots_by_dt()
    # plots_by_net()
    # plots_interlayers()
