import torch
import numpy as np
from lorenz_map import LorenzMap

def get_test_traj(num, delta_t):
    """
    Get Lorenz system object used to compute trajectory of the Lorenz
    with Runge-Kutta method
    """
    if num == 0:
        lor = LorenzMap(delta_t=delta_t)
        init = np.array([0, 1, 1.05])
        return lor, init
    if num == 1:
        lor = LorenzMap(delta_t=delta_t)
        init = np.array([0, -50, 20])
        return lor, init
    if num == 2:
        sigma, rho, beta = 5, 10, 1
        lor = LorenzMap(sigma, rho, beta, delta_t)
        init = np.array([5.0, 0, 10])
        return lor, init
    if num == 3:
        sigma, rho, beta = 8, 25, 3
        lor = LorenzMap(sigma, rho, beta, delta_t)
        init = np.array([5.0, 0, 10])
        return lor, init
    if num == 4:
        sigma, rho, beta = 5, 30, 2.5
        lor = LorenzMap(sigma, rho, beta, delta_t)
        init = np.array([5.0, 0, 10])
        return lor, init
    if num == 5:
        sigma, rho, beta = 20, 50, 10
        lor = LorenzMap(sigma, rho, beta, delta_t)
        init = np.array([5.0, 5, 10])
        return lor, init

def get_model(name, device):
    """
    Get model loaded with its state after training
    Need to manually enter possibilities to return the correct model
    name should correspond to a directory containing the state of a trained model
    """
    seq = torch.load('./'+name+'/'+name+'_trained.pt')
    seq = seq.double().to(device)
    seq = seq.eval()
    seq.load_state_dict(torch.load('./'+name+'/'+name+'_training_state.pt'))
    return seq
