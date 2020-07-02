import torch
from pytorch_expm import torch_expm

def get_alpha2(grad1, grad2):
    grad1 = torch.cat([torch.flatten(grad) for grad in grad1])
    grad2 = torch.cat([torch.flatten(grad) for grad in grad2])
    v1v1 = torch.t(grad1) @ grad1
    v1v2 = torch.t(grad1) @ grad2
    v2v2 = torch.t(grad2) @ grad2
    if v1v2 >= v2v2:
        return [0, 1]
    elif v1v2 >= v1v1:
        return [1, 0]
    else:
        grad2m1 = grad2 - grad1
        res = (torch.t(grad2m1) @ grad2) / (torch.norm(grad2m1) ** 2)
        return [res, 1 - res]

def jacobians_data(lorenz, data, delta_t):
    """
    True jacobian
    """
    sigma, rho, beta = lorenz.sigma, lorenz.rho, lorenz.beta
    # print(data.size())
    jacob_analyt = torch.zeros(data.size()[0], 3, 3)
    const = torch.tensor([[ -sigma, sigma,       0],
                          [rho,      -1,     0],
                          [        0,      0, -beta]], dtype=torch.double)
    # print(const)
    for i in range(len(data)):
        jacob_analyt[i] = torch.add(torch.tensor([[0, 0, 0],
                                                  [-data[i, 0, 2], 0, -data[i, 0, 0]],
                                                  [data[i, 0, 1], data[i, 0, 0], 0]], dtype=torch.double), const)
    jacob_analyt = torch_expm(jacob_analyt * delta_t)
    return jacob_analyt
