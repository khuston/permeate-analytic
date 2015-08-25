# Example usage 
import numpy as np
import matplotlib.pyplot as plt
from slab import Slab

slab1 = Slab(Bi=np.inf, L=0.02, D=9e-7, c_L=2., c_inf=0., num_eigv=30)
x     = np.linspace(0,0.02,1000)
for t in np.logspace(-2,2,100):
   plt.plot(x,slab1.evaluate(x,t))

plt.show()
