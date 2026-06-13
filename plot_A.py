import matplotlib.pyplot as plt
import numpy as np

def plot_A(eps):
    """
    plot the function A(x) = 1/( 2+ cos(2*pi*x/eps))
    """
    n = len(eps)
    x = np.linspace(0, 1, 1000)

    for i in range(n) :
        y = 1/( 2+ np.cos(2 + np.pi*x/eps[i]))
        plt.subplot(1, n, i+1)
        plt.plot(x, y)
        plt.title("Graphe de A")
        plt.grid(True)
    plt.show()
    return

plot_A([0.1, 0.2, 0.3])
