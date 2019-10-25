import matplotlib.pyplot as plt
import numpy as np


def main():
    x = np.array(range(100))
    y = np.zeros(100)
    for i in range(len(x)):
        y[i] = pow(x[i], 0.5)

    plt.figure()
    plt.plot(x, 1 - y / 10, label='coefficients', linestyle='--', marker='o')
    plt.legend()
    plt.xlabel("Number of kernels")
    plt.ylabel("Coefficient")
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
