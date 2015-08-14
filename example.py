# Example usage 
import numpy as np
import matplotlib.pyplot as plt
from AnalyticSlabDiffusion import Slab

slab = Slab(Bi=np.inf, L=0.02, D=9e-7, c_L=2., c_inf=0., num_eigv=1000)
x = np.linspace(0,0.02)
for t in [1,3,10]:
   plt.plot(x,slab.evaluate(x,t))
plt.show()
