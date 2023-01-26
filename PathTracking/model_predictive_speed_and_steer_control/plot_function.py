import numpy as np
import matplotlib.pyplot as plt


# evenly sampled time at 200ms intervals
t = np.arange(-46, 46, 0.2)
fct = (1/(45-abs(t+0.001)))**2
fct2 = np.exp(abs(t))

# red dashes, blue squares and green triangles
plt.plot(t, fct, 'r-')
# plt.plot(t, fct2, 'b--')
plt.show()