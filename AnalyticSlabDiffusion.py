# Diffusion through a slab with mixed boundary conditions and initial
# temperature/concentration equal to bulk phase contacting the non-Dirichlet
# boundary.
#
# Written by Kyle Huston on Monday, July 13, 2015
#
# This solution comes from the key of Prof. Charles Monroe's final exam
# for his transport class at the University of Michigan.
# It is a spectral solution arising from non-dimensionalization, separation of
# variables into two Sturm-Liouville problems coupled by an eigenvalue, and
# application of the dimensionless boundary conditions and initial conditions.

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi, exp
from scipy import optimize

# x is dimensionless position (Position) / (Length)
# t is dimensionless time (Diffusion coefficient * time) / (Length ** 2)
# Bi is the Biot number (Tranfser coefficient * Length) / (Diffusion coefficient ** 2)
#  Note: setting Bi number to infinity (e.g. with np.inf) makes transfer at the x=0
#        boundary instantaneous, and therefore makes the x=0 temperature/concentration
#        constant
# k indexes the basis functions X(x,Bi,k)*T(t,k)
# eigv(k) is the kth eigenvalue
# X is the position-dependent factor of the basis function
# T is the time-dependent factor of the basis function
# A(Bi,k) is the coefficient for the kth basis function

# residual (i.e. left-hand side of equation f(eigv(k)) = 0) to calculate eigenvalues
def eigv_residual(eigv_k,k,Bi):
    return eigv_k + np.arctan(eigv_k/Bi) - k*pi

def T(t,k):
    return exp(-eigv[k]**2*t)

def X(x,Bi,k):
    return sin(eigv[k]*x)/eigv[k] + cos(eigv[k]*x)/Bi

def A(Bi,k):
    x = np.linspace(0,1,100)
    return -np.trapz(Theta_ss(x,Bi)*X(x,Bi,k),x)/np.trapz(X(x,Bi,k)**2,x)

# steady-state solution
def Theta_ss(x,Bi):
    if Bi < np.inf:
        return (1+Bi*x)/(1+Bi)
    else:
        return x

# time-dependent part of solution
def Theta_t(x,t,Bi):
    return np.sum(np.array([ A(Bi,k)*T(t,k)*X(x,Bi,k) for k in range(1,999)]),axis=0)

# full solution
def Theta(x,t,Bi):
    return Theta_ss(x,Bi)+Theta_t(x,t,Bi)

# ------------------------ #
# Plot solutions
Bi = np.inf
eigv = [optimize.newton(eigv_residual, x0=k, args=(k,Bi)) for k in range(0,1000)]
t = 0.
x = np.linspace(0.,1.,100)

for t in np.logspace(-3,1):
    plt.plot(x,Theta(x,t,Bi))
    plt.xlim([0,1])
    plt.ylim([0,1])
plt.show()
