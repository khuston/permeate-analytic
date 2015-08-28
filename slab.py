"""
    Diffusion through a slab with mixed boundary conditions and initial
    temperature/concentration equal to bulk phase contacting the non-Dirichlet
    boundary.
                                                                               
    Written by Kyle Huston on Monday, July 13, 2015
    Last updated on Friday, August 28, 2015

    This solution comes from the key of Prof. Charles Monroe's final exam
    for his transport class at the University of Michigan.
    It is a spectral solution arising from non-dimensionalization, separation of
    variables into two Sturm-Liouville problems coupled by an eigenvalue, and
    application of the dimensionless boundary conditions and initial conditions.
                                                                                 
     (A) Neumann boundary condition at x = 0
     D (dc/dx)|0 = h(c(0) - c_inf)
                                                                                 
     (B) Dirichlet boundary condition at x = L
     c(L) = c_L
                                                                                 
     (C) Governing equation
     dc/dt = D d^2c/dx^2
     \frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}
                                                                                 
     (I) Initial condition
     c(x) = c_inf
                                                                                 
     Schematic of steady-state solution to diffusion with large Biot number 
               
                       (A)        (C)      (B)
                        |                 //|
                        |             //////|
            c_inf  c(0) |        ///////////| c(L)
                        |    ///////////////|
                        |///////////////////|
                      x = 0               x = L

    x is dimensionless position (Position) / (Length)
    t is dimensionless time (Diffusion coefficient * time) / (Length ** 2)
    Bi is the Biot number (Tranfser coefficient * Length) / (Diffusion coefficient)
     Note: setting Bi number to infinity (e.g. with np.inf) makes transfer at the x=0
           boundary instantaneous, and therefore makes the x=0 temperature/concentration
           constant
    k indexes the basis functions X(x,Bi,k)*T(t,k)
    eigv(k) is the kth eigenvalue
    X is the position-dependent factor of the basis function
    T is the time-dependent factor of the basis function
    A(Bi,k) is the coefficient for the kth basis function
"""

import numpy as np
from numpy import sin, cos, pi, exp, arctan, inf as infinity
from scipy import optimize
from scipy.special import erf

# residual (i.e. left-hand side of equation f(eigv(k)) = 0) to calculate eigenvalues
def eigv_residual(eigv_k,k,Bi):
    if Bi == 0:
        return eigv_k + pi/2. - k*pi
    else:
        return eigv_k + arctan(eigv_k/Bi) - k*pi

def T(t,eigv_k):
    return exp(-eigv_k**2*t)

def X(x,Bi,eigv_k):
    return sin(eigv_k*x)/eigv_k + cos(eigv_k*x)/Bi

def A(Bi,eigv_k):
    x = np.linspace(0,1,100)
    return -np.trapz(Theta_ss(x,Bi)*X(x,Bi,eigv_k),x)/np.trapz(X(x,Bi,eigv_k)**2,x)

# NOTE: The noflux functions are only to be used _together_ and when Bi == 0
#       because Biot number cancels out in no flux case
def X_noflux(x,eigv_k):
    return cos(eigv_k*x)

def A_noflux(eigv_k):
    x = np.linspace(0,1,100)
    return -np.trapz(X_noflux(x,eigv_k),x)/np.trapz(X_noflux(x,eigv_k)**2,x)

# steady-state solution
def Theta_ss(x,Bi):
    if Bi < np.inf:
        return (1+Bi*x)/(1+Bi)
    else:
        return x

# time-dependent part of solution
def Theta_t(x,t,Bi,eigv):
    if Bi == 0:
        return np.sum(np.array([ A_noflux(eigv[k])*T(t,eigv[k])*X_noflux(x,eigv[k]) for k in range(1,len(eigv)-1)]),axis=0)
    else:
        return np.sum(np.array([ A(Bi,eigv[k])*T(t,eigv[k])*X(x,Bi,eigv[k]) for k in range(1,len(eigv)-1)]),axis=0)

# full solution
def Theta(x,t,Bi,eigv):
    return Theta_ss(x,Bi)+Theta_t(x,t,Bi,eigv)

def Theta_short(x,t,D):
    return -erf(x/(2*np.sqrt(D*t)))+1.

class Slab:
    """
        Slab class for calculating concentration profile for some x,t in a slab.

        Biot number, length, diffusion coefficient, and boundary conditions are
        set at initialization before eigenvalues are calculated.

        These quantities are dimensional. The evaluate method converts x,t to 
        dimensionless x/L, tD/L**2 and returns a dimensional concentration as
        (c_L - c_inf)*Theta + c_inf where Theta is a dimensionless concentration.

        Warning: evaluate(x,t) does not re-calculate eigenvalues, so if you manually change
        the Biot number, then you must call _calculate_eigv() before evaluating again, or else
        your answer will be wrong.
    """
    def __init__(self,Bi,L,D,c_L,c_inf,num_eigv=100):
        self._Bi = Bi
        self._L = L
        self._D = D
        self._c_L = c_L
        self._c_inf = c_inf
        self._num_eigv = num_eigv
        self._calculate_eigv()
        
    def _calculate_eigv(self):
        self._eigv = [optimize.newton(eigv_residual, x0=k, args=(k,self._Bi)) for k in range(self._num_eigv)]

    def evaluate(self,x,t):
        """
            evaluate takes dimensional x and t. It converts them to dimensionless x/L and tD/L**2
            and returns a dimensional concentration based on the c_L and c_inf provided at initialization.
        """
        L = self._L
        D = self._D
        Bi = self._Bi
        c_L = self._c_L
        c_inf = self._c_inf
        eigv = self._eigv
        if t*D/L**2 > 0.001:
            return Theta(x/L,t*D/L**2,Bi,eigv)*(c_L-c_inf)+c_inf
        else:
            return (Theta_short(x,t,D)*(c_L-c_inf)+c_inf)[::-1]
