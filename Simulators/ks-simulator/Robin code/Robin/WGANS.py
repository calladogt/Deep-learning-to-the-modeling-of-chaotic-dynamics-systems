import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import utils_saving as us
import utils_plot as up
import gan_networks as gan
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


parser = argparse.ArgumentParser('GAN expe')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=50000)
parser.add_argument('--batch_time', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--rnn_size', type=int, default=2000)
parser.add_argument('--niters', type=int, default=2500)
parser.add_argument('--test_freq', type=int, default=5)
parser.add_argument('--viz', type=eval, default=True, choices=[True, False])
parser.add_argument('--restart', type=eval, default=False, choices=[True, False])
parser.add_argument('--begin_epoch', type=int, default=0)
parser.add_argument('--pretrain', type=eval, default=False)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

torch.set_default_tensor_type('torch.DoubleTensor')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(device)

# def makedirs(dirname):
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)

def lorenz(t, v, c):
    x, y, z = v
    sigma, rho, beta = c
    dot_x = sigma * (-x + y)
    dot_y = x * (rho - z) - y
    dot_z = x * y - beta * z
    return [dot_x, dot_y, dot_z]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def get_batch(input, t):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_t = t[:args.batch_time]
    batch_y = torch.stack([input[s + i] for i in range(args.batch_time)], dim=0)
    return batch_t, batch_y

def get_ic(input):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.rnn_size, dtype=np.int64), args.batch_size, replace=False))
    batch_y = torch.stack([input[s + i] for i in range(args.rnn_size)], dim=0)
    return batch_y

def get_batch_test(input):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), 1, replace=False))
    batch_y_truth = torch.stack([input[s + i] for i in range(args.batch_time)], dim=0)
    return s, batch_y_truth, batch_y_truth[:args.rnn_size, :, :]

def calc_gradient_penalty(Dis, real_data, fake_data):
    alpha = torch.ones_like(real_data).to(device) * torch.rand(args.batch_size).reshape(-1, 1, 1, 1).to(device)

    interpolates = (alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())).to(device)
    interpolates.requires_grad = True

    dis_interpolates = Dis(interpolates)

    gradients = autograd.grad(outputs=dis_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(dis_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

num_pred = 0

def gen_pretraining(filename, Generator, inputs, nb_epochs, saved=False):
    inp = inputs[:-1, :].unsqueeze(1)
    target = inputs[1:, :].unsqueeze(1)

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(Generator.parameters(),  lr=0.1)

    if saved:
        Generator.load_state_dict(torch.load('./'+filename+'/pretrain/'+filename+'training.pt'))
        optimizer.load_state_dict(torch.load('./'+filename+'/pretrain/'+ filename + 'optimizer.pt'))


    else:
        us.dir_network(filename+"/pretrain")

        def closure():
            optimizer.zero_grad()
            out = Generator.pretrain(inp)

            loss = criterion(out, target)
            print('loss:', loss.item())
            us.save_trainlog(filename, 'loss:' + str(loss.item()))
            loss.backward()

            return loss

        for i in range(1, nb_epochs + 1):
            optimizer.step(closure)
            print('EPOCH: ', i)
            us.save_trainlog(filename, "Epoch " + str(i))
            torch.save(Generator.state_dict(), './'+filename+'/pretrain/'+filename+'training.pt')
            torch.save(optimizer.state_dict(), './'+filename+'/pretrain/'+filename+'optimizer.pt')


if __name__ == '__main__':

    # set random seed to 0
    np.random.seed(0)

    # GAN9 with bigger generator

    filename = "GAN9"
    filename_saved = "GAN9"
    us.dir_network(filename)
    Gen = gan.Generator3(args.batch_time, 50).double().to(device)           #Generator
    Dis = gan.Discriminator2().double().to(device)     #Discriminator
    Gen.apply(weights_init)
    Dis.apply(weights_init)
    d_optimizer = optim.Adam(Dis.parameters(), lr=1e-4, betas=(0, 0.9))
    g_optimizer = optim.Adam(Gen.parameters(), lr=1e-4, betas=(0, 0.9))

    if args.restart:
        Gen.load_state_dict(torch.load('./'+filename_saved+'/Gen_training.pt'))
        g_optimizer.load_state_dict(torch.load('./'+filename_saved+'/Gen_optimizer.pt'))
        Dis.load_state_dict(torch.load('./'+filename_saved+'/Dis_training.pt'))
        d_optimizer.load_state_dict(torch.load('./'+filename_saved+'/Dis_optimizer.pt'))

    dfe, dre, ge = 0, 0, 40

    #begin to train
    t = np.linspace(0., args.data_size*25*0.001, args.data_size*25)
    c = [10, 28, 8/3]
    vinit = [-5.76, 2.27, 32.82]
    data = solve_ivp(lambda t, v: lorenz(t, v, c), [t[0], t[-1]], vinit, t_eval=t)
    t = torch.from_numpy(t).to(device)
    t = t[::25]
    input = torch.from_numpy(np.transpose(np.asarray(data.y), (1, 0))).to(device)
    input = input[::25]
    # input = [seq, features]
    if args.viz:
        # up.plot_traj_GAN(input[:args.batch_time], t[:args.batch_time], "Target_0_to_"+str(args.batch_time), filename)
        up.plot_traj_GAN(input, t, "Target", filename)

    LAMBDA = 10

    us.save_settings(filename, "GAN without gradient penalty, parameters with default values\n Generator is same as MLP1_5, with 100 epochs pretraining\n Prediction alone with truth input each 50 time steps\n Lambda for penalty = 10, output of discriminator softmax proba")

    pretrained_network = "GAN9"
    if args.pretrain:
        gen_pretraining(pretrained_network, Gen, input, 1, saved=True)

    one = torch.Tensor([1]).to(device)
    mone = (one * -1).to(device)

    for i in range(args.begin_epoch, args.niters):
        for d_index in range(5):
            batch_t, batch_y = get_batch(input, t)
            # batch_y  = [seq, batch,  features]
            # first train discriminant
            Dis.zero_grad()

            # on real data
            batch_y = batch_y.permute(1, 0, 2).unsqueeze(1)
            # batch_y  = [batch, channel, seq,  features]
            real_decision = Dis(batch_y) # (1 or 0)
            # print("Real decision", real_decision.size(), real_decision)
            real_loss = real_decision.mean().to(device)
            # real_loss = real_decision.sum().to(device)
            real_loss.backward(mone)

            # on fake data
            batch_y0 = get_ic(input)
            # batch_y0 = [seq, batch,  features]
            fake_data = Gen(batch_y0).detach()
            # fake_data = [seq, batch,  features]
            fake_data = fake_data.unsqueeze(1).permute(2, 1, 0, 3)
            # fake_data = [batch, channel, seq,  features]
            fake_decision = Dis(fake_data)
            # print("Fake decision", fake_decision.size(), fake_decision)
            fake_loss = fake_decision.mean().to(device)
            # fake_loss = fake_decision.sum().to(device)
            fake_loss.backward(one)

            gradient_penalty = LAMBDA * calc_gradient_penalty(Dis, batch_y, fake_data)
            # gradient_penalty.backward(one)

            # Full_loss = fake_loss - real_loss + gradient_penalty
            Full_loss = fake_loss - real_loss
            # Full_loss.backward()
            d_optimizer.step()

            dre, dfe = real_loss.item(), fake_loss.item()
            gp = gradient_penalty.item()

            dfl = Full_loss.item()

        for g_index in range(1):
            # then train generator
            Gen.zero_grad()

            batch_y0 = get_ic(input)
            # batch_y0 = [seq, batch,  features]
            gen_data = Gen(batch_y0)
            # gen_data = [seq, batch,  features]
            gen_data = gen_data.unsqueeze(1).permute(2, 1, 0, 3)
            # gen_data = [batch, channel, seq,  features]
            gen_decision = Dis(gen_data)
            gen_loss = gen_decision.mean().to(device)
            # gen_loss = gen_decision.sum().to(device)
            gen_loss.backward(mone)

            g_optimizer.step()
            ge = gen_loss.item()

        if i% args.test_freq == 0:
            print('Epoch {:04d}: D ({:.2e} real_val, {:.2e} fake_val, {:.2e} full loss) G ({:.2e} gen_val)'.format(i, dre, dfe, dfl, ge))
            # print('Epoch {:04d}: D ({:.2e} real_err, {:.2e} fake_err, {:.2e} gradient penalty, {:.2e} full loss) G ({:.2e} err)'.format(i, dre, dfe, gp, dfl, ge))
            us.save_trainlog(filename, 'Epoch {:04d}: D ({:.2e} real_err, {:.2e} fake_err, {:.2e} full loss) G ({:.2e} err)'.format(i, dfl, dre, dfe, ge))
            torch.save(Gen.state_dict(), './'+filename+'/Gen_training.pt')
            torch.save(g_optimizer.state_dict(), './'+filename+'/Gen_optimizer.pt')
            torch.save(Dis.state_dict(), './'+filename+'/Dis_training.pt')
            torch.save(d_optimizer.state_dict(), './'+filename+'/Dis_optimizer.pt')


            with torch.no_grad():
                # batch_y0 = get_ic(input)
                # gen_data = Gen(batch_y0[:, :1, :])
                time_begin, truth, test_data = get_batch_test(input)
                gen_data = Gen(test_data)
                if args.viz:
                    up.plot_traj_GAN_test(truth.squeeze(1), gen_data.squeeze(1), t[time_begin:time_begin+args.batch_time], "Ep_"+str(i), filename)


