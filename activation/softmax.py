import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)

def plot(x, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    x = np.array([2, 3, 4])
    soft = softmax(x)
    plot(x, soft, r"$f(x_i)=\frac{e^{x_i}}{\Sigma_{j=0}^ke^{x_j}}$")
