import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_hard(x):
    return np.clip((x+1)/2, 0, 1)

def sigmoid_log(x):
    return - np.logaddexp(0, -x)

def plot(x, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-5, 5, 100)
    s1 = sigmoid(x)
    plot(x, s1, r"$sigmoid = \frac{1}{1+e^{-x}}$")

    s2 = sigmoid_hard(x)
    plot(x, s2, r"$sigmoid_{hard}=\frac{x+1}{2}$")

    s3 = sigmoid_log(x)
    plot(x, s3, r"$log\_sigmoid=log\frac{1}{1+e^{-x}}$")

    plt.plot(x, s1, label=r"$sigmoid = \frac{1}{1+e^{-x}}$")
    plt.plot(x, s2, label=r"$sigmoid_{hard}=\frac{x+1}{2}$")
    plt.title("Sigmoid vs. Sigmoid_hard")
    plt.legend()
    plt.show()