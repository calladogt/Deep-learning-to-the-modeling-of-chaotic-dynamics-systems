import torch
import time
import numpy as np
from numpy.linalg import qr
from lorenz_map import LorenzMap
from scipy.linalg import expm

def stable_le(jacobian, traj, delta_t=1e-3):
    """
    Function which is the most stable with varying delta t
    ____
    Function used for analytical system
    """

    n = len(traj[0])
    w = np.eye(n)
    r = np.zeros(n)
    rs = []
    x = traj[0]

    for i in range(1, len(traj)):
        x_next = traj[i]
        jacob = jacobian(x)
        w_next = np.dot(expm(jacob * delta_t), w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

def lyapunov_exponent(step, jacobian, initial_condition=np.array([-5.76,  2.27,  32.82]), max_it=1000, delta_t=1e-3):
    """
    Computation of Lyapunov exponents of an analytical physical system
    Step is the function of the system that performs an evolution of one time step (related to the delta t)
    Jacobian is the function providing the analytical jacobian of the system
    max_it is a parameter defining of many steps we make
    ____
    Original function, no longer used
    """

    n = len(initial_condition)
    w = np.eye(n)
    rs = []
    x = initial_condition

    for i in range(1, max_it):
        x_next = step(x)
        jacob = jacobian(x)
        w_next = np.dot((np.eye(n) + jacob * delta_t), w)

        w_next, r_next = qr(w_next)

        # qr computation from numpy allows negative values in the diagonal
        # Next three lines to have only positive values in the diagonal
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

def lyapunov_exponent_v2(step, jacobian, initial_condition=np.array([-5.76,  2.27,  32.82]), max_it=1000, delta_t=1e-3):
    """
    Same function as before, but uses the true matrix exponential instead of a first order approximation
    for the computation of w_next
    ____
    Also not used
    """

    n = len(initial_condition)
    w = np.eye(n)
    rs = []
    x = initial_condition

    for i in range(1, max_it):
        x_next = step(x)
        jacob = jacobian(x)
        w_next = np.dot(expm(jacob * delta_t), w)

        w_next, r_next = qr(w_next)
        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

def lyapunov_exponent_fulltraj(jacobian, traj, delta_t=1e-3):

    """
    Test to see whether lyapunov exponents with points computed with Runge Kutta 4th
    order make better lyapunovs exponents
    Results : Worse exponents, mainly for the second value, but more stable for the first
    """

    n = len(traj[0])
    w = np.eye(n)
    r = np.zeros(n)
    rs = []
    x = traj[0]

    for i in range(1, len(traj)):
        x_next = traj[i]
        jacob = jacobian(x)
        w_next = np.dot((np.eye(n) + jacob * delta_t), w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

def lyapunov_exponent_fulltraj_v2(jacobian, traj, delta_t=1e-3):
    """
    Test to see whether lyapunov exponents matrix exponential instead of first order
    interpolation and with points computed with Runge Kutta 4th
    order would make better lyapunovs exponents
    """
    n = len(traj[0])
    w = np.eye(n)
    r = np.zeros(n)
    rs = []
    x = traj[0]

    for i in range(1, len(traj)):
        x_next = traj[i]
        jacob = jacobian(x)
        w_next = np.dot(expm(jacob * delta_t), w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t


def lyapunov_exponent_NN(step, jacobian, initial_condition=np.array([-5.76,  2.27,  32.82]), max_it=1000, delta_t=1e-3):

    """
    Compared to lyapunov exponents on data, lyapunov exponents on the ML model
    doesn't need first order estimation for the true Jacobian which is already given
    ____
    This method doesn't give correct estimation of the exponents because
    the network has to be able to predict alone in the far future
    """
    n = len(initial_condition)
    w = np.eye(n)
    rs = []
    x = initial_condition

    for i in range(1, max_it):
        x_next = step(x)
        x_next = x_next.reshape(-1, 3)

        jacob = jacobian(x)

        w_next = np.dot(jacob, w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0)

def lyapunov_exponent_fulltraj_NN(jacobian, traj, delta_t=1e-3):

    """
    Uses the whole trajectory predicted by the network to compute lyapunov exponents
    Probably should work only if predictions are big enough
    ____
    more than 1e5 steps can give correct LE
    """
    traj = traj.reshape(-1, 3)
    n = len(traj[0])
    w = np.eye(n)
    rs = []
    x = traj[0]

    for i in range(1, len(traj)):
        x_next = traj[i]
        jacob = jacobian(x)

        w_next = np.dot(jacob, w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

def lyapunov_exponent_fulltraj_resNN(jacobian, traj, delta_t=1e-3):

    """
    Uses the whole trajectory predicted by the network to compute lyapunov exponents
    Probably should work only if predictions are big enough
    ____
    more than 1e5 steps can give correct LE
    """
    traj = traj.reshape(-1, 3)
    n = len(traj[0])
    w = np.eye(n)
    rs = []
    x = traj[0]

    for i in range(1, len(traj)):
        x_next = traj[i]
        jacob = jacobian(x)
        w_next = np.dot(expm(jacob * delta_t), w)
        # w_next = np.dot(jacob, w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

def lyapunov_exponent_fulltraj_interNN(jacobian, traj, delta_t=1e-3):

    """
    Compute lyapunov exponents for intermediate layers
    ___
    Used when the size of the values used for computation isn't 3
    """
    traj = traj.reshape(-1, 3)
    n = len(jacobian(traj[0])[0])
    w = np.eye(n)
    rs = []
    x = traj[0]

    for i in range(1, len(traj)):
        x_next = traj[i]
        jacob = jacobian(x)

        w_next = np.dot(jacob, w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        x = x_next
        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

def lyapunov_exponent_fulltraj_lstm(jacobians, delta_t=1e-3):

    """
    Uses the whole trajectory's jacobians on predictions by the network to compute lyapunov exponents
    Used for CNN and LSTM where it is easier to compute jacobians in advance
    ____
    more than 1e5 steps can give correct LE
    """
    n = len(jacobians[0, 0])
    w = np.eye(n)
    rs = []

    for i in range(0, len(jacobians)):
        jacob = jacobians[i]

        w_next = np.dot(jacob, w)
        w_next, r_next = qr(w_next)

        d = np.diag(np.sign(r_next.diagonal()))
        w_next = np.dot(w_next, d)
        r_next = np.dot(d, r_next.diagonal())

        rs.append(r_next)

        w = w_next

    return np.mean(np.log(rs), axis=0) / delta_t

if __name__ == '__main__':
    """
    Some tests to see the quality of estimations
    True values are
    0.905+- 0.005, 0.0, -14.57+-0.01 (found in the book An introduction to dynamical systems)
    """

    file = open("test_lyapunovs.txt", "w+")

    for nb_iter in [10000, 100000, 1000000]:#[1000, 10000, 100000, 1000000, 10000000]:
        file.write("nombre d'iterations : " + str(nb_iter)+ "\n")
        print("Number of iterations to compute exponents :", nb_iter)
        for delta_t in [1e-3, 5e-3, 1e-2]:#[1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]:
            file.write("Delta t to compute exponents :" + str(delta_t)+ "\n")
            print("Delta t :", delta_t)
            LORENZ_MAP = LorenzMap(delta_t=delta_t)
            INIT = np.array([0, 1, 1.05])
            t = time.time()
            lyap = lyapunov_exponent(LORENZ_MAP.step, LORENZ_MAP.jacobian, initial_condition=INIT, max_it=nb_iter, delta_t=delta_t)
            file.write("Exponents with formula : "+ str(lyap) + " in "+ str(time.time() - t)+" sec\n")
            print("Exposants formule :", lyap, "in "+ str(time.time() - t)+" sec")

            t = time.time()
            lyap1 = lyapunov_exponent_v2(LORENZ_MAP.step, LORENZ_MAP.jacobian, initial_condition=INIT, max_it=nb_iter, delta_t=delta_t)
            file.write("Exponents with formula v2 : "+ str(lyap1) +" in "+ str(time.time() - t)+" sec\n")
            print("Exposants formule v2 :", lyap1, "in "+ str(time.time() - t)+" sec")

            t = time.time()
            traj = LORENZ_MAP.full_traj(nb_iter, INIT)
            lyap2 = lyapunov_exponent_fulltraj(LORENZ_MAP.jacobian, traj, delta_t=delta_t)
            file.write("Exponents with trajectory by integration : "+ str(lyap2) + " in "+ str(time.time() - t)+" sec\n")
            print("Exposants traj integree :", lyap2, "in "+ str(time.time() - t)+" sec")

            t = time.time()
            lyap3 = lyapunov_exponent_fulltraj_v2(LORENZ_MAP.jacobian, traj, delta_t=delta_t)
            file.write("Exponents with trajectory by integration v2 : "+ str(lyap3) + " in "+ str(time.time() - t)+" sec\n")
            print("Exposants traj integree v2 :", lyap3, "in "+ str(time.time() - t)+" sec")

    file.close()
