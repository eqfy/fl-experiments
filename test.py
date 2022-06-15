from matplotlib import pyplot as plt
import numpy as np


a = np.random.lognormal(0, 1, 100)
a = np.sort(a)
print(a)
plt.plot(a)
plt.yscale('log')
plt.show()
