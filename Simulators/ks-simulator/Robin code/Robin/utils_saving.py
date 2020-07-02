import os

def dir_network(filename, plot_dir="plots", logfile="trainlog.txt", lyapfile="lyapunovs.txt"):
    path_dir = filename
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    if not plot_dir == None and not os.path.exists(path_dir+"/"+plot_dir):
        os.mkdir(path_dir+"/"+plot_dir)
    file_log = open("./"+filename+"/"+logfile, "w")
    file_log.close()
    file_lyap = open("./"+filename+"/"+lyapfile, "w")
    file_lyap.close()

def dir_test(filename):
    path_dir = filename
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

def save_settings(filename, content, step=None):
    if step is None:
        file = open("./"+filename+"/settings.txt", "w+")
    else:
        file = open("./"+filename+"/settings_step"+str(step)+".txt", "w+")
    file.write(content)
    file.close()

def dir_jacobians(filename, jacob_dir="jacobians"):
    path_dir = "./"+filename +"/"+jacob_dir
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)

def save_trainlog(filename, loss_txt):
    file = open(filename, "a+")
    file.write(loss_txt + "\n")
    file.close()

def save_lyap(filename, lyap_txt):
    file = open(filename, "a+")
    file.write(lyap_txt + "\n")
    file.close()
