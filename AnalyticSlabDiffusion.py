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
#
#  (A) Neumann boundary condition at x = 0
#  D (dc/dx)|0 = h(c(0) - c_inf)
#
#  (B) Dirichlet boundary condition at x = L
#  c(L) = c_L
#
#  (C) Governing equation
#  dc/dt = D d^2c/dx^2
#  \frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}
#
#  (I) Initial condition
#  c(x) = c_inf
#
#  Schematic of steady-state solution to diffusion with large Biot number 
#                
#                        (A)        (C)      (B)
#                         |                 //|
#                         |             //////|
#             c_inf  c(0) |        ///////////| c(L)
#                         |    ///////////////|
#                         |///////////////////|
#                       x = 0               x = L

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi, exp
from scipy import optimize

# x is dimensionless position (Position) / (Length)
# t is dimensionless time (Diffusion coefficient * time) / (Length ** 2)
# Bi is the Biot number (Tranfser coefficient * Length) / (Diffusion coefficient)
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
# Plot solution

# Adjustable parameters
L = 0.02                      # cm            Slab width
D = 9e-7                      # cm**2/s       Diffusion coefficient
h = np.inf                    # cm/s          Mass transfer coefficient at x=0 wall
c_inf = 0.                    # mole/cm**3    Concentration at x=-infinity
c_L = 2.                      # mole/cm**3    Concentration at x=L

# Create spatial and temporal variables 
x = np.linspace(0.,L,100)
t = np.logspace(-3,2)

# Create dimensionless variables and eigenvalues
Bi = h*L/D
eigv = [optimize.newton(eigv_residual, x0=k, args=(k,Bi)) for k in range(0,1000)]
x_dimless = x/L
t_dimless = t*D/L**2

# Plot solution at each time
for each_t in t_dimless:
    plt.plot(x,Theta(x_dimless,each_t,Bi)*(c_L-c_inf)+c_inf)
    plt.xlim([0,L])
    plt.ylim([c_inf,c_L])
plt.show()
