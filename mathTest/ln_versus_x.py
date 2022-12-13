"""
ln(x+1)与x的关系:
x > ln(x+1), when x > 0
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.log(x+1)

plt.plot(x, x, color="g", label=r"$y=x$")
plt.plot(x, y, color="b", label=r"$y=ln(x)$")
plt.legend()
plt.xlim(0, 11)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()
