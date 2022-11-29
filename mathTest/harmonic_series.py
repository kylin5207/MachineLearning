"""
调和级数
Sn=1+1/2+1/3+...+1/n  n->∞

- 高数书上近似值为ln(n+1)
- 但是发现更近似的结果 ln(n+gamma)  其中gamma近似0.577（参考https://en.wikipedia.org/wiki/Harmonic_series_(mathematics)）
"""

import numpy as np
import matplotlib.pyplot as plt

res = [1]
n = 100000

for i in range(2, n+1):
    res.append(res[-1]+1/i)

x = np.arange(1, n+1)
plt.scatter(x, res, s=20, alpha=0.2, c='r')
plt.plot(x, res, linewidth=2, label=r"$\sum_{i=1}^{n} \frac{1}{i} $")
plt.xlabel(r"$x$")
plt.title(r"$\sum_{i=1}^{n} \frac{1}{i} $")
print(np.log(n+1))
plt.axhline(np.log(n+1), 0, n, color='black', linestyle='--', label=r'$ln(n)+\gamma$')
plt.axhline(np.log(n)+0.577, 0, n, color='green', linestyle='-.', label=r'$ln(n+1)$')
plt.legend()
plt.show()
