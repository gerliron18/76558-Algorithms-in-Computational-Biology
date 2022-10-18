

from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt


a, b = 10, 10
x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 100)

fig, ax = plt.subplots(1, 1)
ax.plot(x, beta.pdf(x, a, b), 'b-', lw=5, alpha=0.6, label='beta pdf')
plt.xlim(0, 1)
plt.show()


_, ax = plt.subplots(1, 1)
ax.plot(x, beta.cdf(x, a, b), 'b-', lw=5, alpha=0.6, label='beta pdf')
plt.xlim(0, 1)
plt.show()



