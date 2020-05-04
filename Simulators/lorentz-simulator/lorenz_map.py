from scipy import integrate
from numpy import *

import numpy as np
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

class LorenzMap:
    """
    LorenzMap in 3D (x,y,z)
    Default values: 
    sigma=10, rho=28, and beta=8/3
    delta_t = 1e-3

    The main method is full_traj which generates the trajectory
    """

    def __init__(_, sigma=10, rho=28, beta=8 / 3, delta_t=1e-3):
        _.sigma, _.rho, _.beta = sigma, rho, beta
        _.delta_t = delta_t

    def v_eq(_, t=None, v=None):
        x, y, z = v[0], v[1], v[2]
        dot_x = _.sigma * (-x + y)
        dot_y = x * (_.rho - z) - y
        dot_z = x * y - _.beta * z
        return array([dot_x, dot_y, dot_z])

    def step(_, v):
        return v + _.v_eq(v=v) * _.delta_t

    def jacobian(_, v):
        x, y, z = v[0], v[1], v[2]
        res = array([[ -_.sigma, _.sigma,       0],
                    [_.rho - z,      -1,      -x],
                    [        y,       x, -_.beta]])
        return res

    
    def full_traj(_, nb_steps, init_pos):
        """
        Generate the trajectory.
        - nb_steps : the number of steps of generation
        - init pos : the starting point, it's a numpy array in 3D 

        Returns: a numpy array of size (nb_steps,3) 
        """
        t = np.linspace(0, nb_steps * _.delta_t, nb_steps)
        f = solve_ivp(_.v_eq, [0, nb_steps * _.delta_t], init_pos, method='RK45', t_eval=t)
        return np.moveaxis(f.y, -1, 0)



# Static method 
def plot_traj(data, filename=None):
    """Plot the result of the simulation. 
    -If the filename is set to None
    (default), the figure is not saved and you can see it.
    -  data is a numpy array of size (nb_steps,3)
    
    If you want to plot only the 100 first points: 
    plot_traj(res[:100])
    """
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    if filename is not None: 
        plt.savefig(filename, format="pdf")
    else: 
        plt.show()


            
def simpleTest():
    gen = LorenzMap(delta_t=1e-2)
    res = gen.full_traj(init_pos=np.ones(3)*0.01, nb_steps=10000)
    plot_traj(res)

    
    
def main():
    simpleTest()

if __name__ == "__main__":
    main()

