# Research internship
## Deep-learning pour la modélisation de systèmes dynamiques chaotiques

I will describe my research internship here. The research subject is "Deep learning to modelization of chaotic dynamics systems".

For instance, I will separate 2 folders of files: articles and codes. In the article folder, I will put all articles that make sense to read and study about my research project. In codes, I will store all codes made during my internship.

Let's study some interesting topics for the project.

### Chaotic systems

Here I will do some revision about this subject. Let's see the conditions to have a chaotic system and how to characterize this kind of system.

#### Conditions required for chaos
As described in the "Dynamical systems and Introduction to Chaos" course (stored in the Article folder), there are two conditions required for a physical system to exhibit chaotic behavior:
1) be nonlinear;
2) possess at least three degrees of freedom.

Remembering these concepts:


* Nonlinear system: A system is nonlinear as soon as its governing equations contain a nonlinear term of $~X$.


* Degrees of freedom: We can see the number of degrees of freedom by the phase space of the system. As we know, the space phase is a way to follow the dynamical evolution of a system, i.e., it shows the evolution of the vector X in its vector space. Its evolution is described by an ensemble of n differential equations associated with initial conditions:

$dX/dt = F(X)$, with an initial condition to X(t=0).

The number of degrees of freedom is the dimension of the vector space necessary to describe the dynamic of the system. For example, in the case of a simple harmonic oscillator, its degrees of freedom is two: we need the position p and the impulsion q to describe completely the system and its trajectory.

#### Chacterizing the chaos

There are some ways to characterize chaos. Here, I will present three simple methods to see if the system is chaotic:

1) Power Spectrum:
A simple way to characterize chaos consists in performing a Fourier
spectrum of the temporal evolution of one variable of the system.
The trajectories of a regular Hamiltonian system is the composition of oscillations each having a pulsation wi, i.e, its power spectrum is a soft spectrum with many harmonics peaks. In the case of a chaotic system, its spectrum is without well-defined frequency peaks but rather broad band noise.

2) Sensitivity to initial conditions:
A more straightforward way to demonstrate that a trajectory is chaotic, consists in measuring it degree of non-predictability. Two experiments starting from rigorously the same initial conditions, evolve with exactly the same trajectory5. But if the initial conditions are not rigorously alike, the distance separating them in phase space will evolve very differently if the trajectory is regular or chaotic:

* For regular trajectories, the two systems with slightly different initial conditions will separate in phase space with a distance growing LINEARLY in time on average.

* For chaotic trajectories, two systems arising from the two different initial conditions are well correlated at the beginning, but they quickly lead to strongly different leading to a complete loss of correlation after a few seconds. In this case, the distance between the two trajectories increases EXPONENTIALLY with time. However, this evolution is not homogeneous in time, but this is only true on average.

3) Stochasticity criterion:
In some systems, we can also see the chaotic behavior with the stochasticity criterion. If a system, apparently regular, has two resonance points and their respective resonance trajectories around these points in the phase space, a possible chaotic movement happens, according to this criterion, when there is an overlapping between the two trajectories around their harmonics. An example of this is the Lorenz attractor. Depending on the parameters chosen for the equation, we have a chaotic behavior of the system in which the trajectories around the resonance points are overlapped and, thus, the system does not know exactly which resonance point it should follow and changes from one to the another chaotically.
