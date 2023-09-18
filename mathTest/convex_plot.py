"""
凸函数 f(x)=e^x

"""

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x = torch.linspace(0, 2, 100)
y = torch.exp(x)

# get two points
point1_x, point1_y = x[20], y[20]
point2_x, point2_y = x[60], y[60]

ef_x, ef_y = (point1_x + point2_x)/2, (point1_y +point2_y)/2
fe_x = (point1_x + point2_x)/2
fe_y = torch.exp(fe_x)

plt.plot(x, y, color='k', label=r"$y=log(x)$")
plt.plot([point1_x, point2_x], [point1_y, point2_y], color='red', linestyle='--')

plt.scatter(point1_x, point1_y, color='green', s=60)
plt.scatter(point2_x, point2_y, color='green', s=60)
plt.scatter(ef_x, ef_y, color='orange', s=60, edgecolors='k', label=r"$E[log(x)] = \frac{e^{x_1}}{2}+\frac{e^{x_2}}{2}$")
plt.scatter(fe_x, fe_y, color='yellow', s=60, edgecolors='k', label=r"$log[E(x)] = e^{\frac{x_1+x_2}{2}}$")
plt.axvline(x=point1_x, color='gray', linestyle='--')
plt.axvline(x=point2_x, color='gray', linestyle='--')

plt.xlabel(r"$x$")
plt.ylabel(r"$e^x$")
plt.title("convex Jensen inequality")
plt.legend()
plt.show()

