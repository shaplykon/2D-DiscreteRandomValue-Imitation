import matplotlib.pyplot as plt
from constants import quantiles
import numpy as np


def show_histogram(values, component_name):
    plt.hist(values, color='Blue')
    plt.title("Histogram of " + component_name + " component of DRV")
    plt.xlabel("Value")
    plt.ylabel("Amount")
    plt.legend((component_name + " component", ''))
    plt.show()


def show_common_histogram(X, Y):
    hist, x_edges, y_edges = np.histogram2d(X, Y)
    hist = hist.T
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.pcolormesh(X, Y, hist)
    plt.show()


def show_deviation_dependence_plot(x_deviation_lengths, y_deviation_lengths):
    plt.title(f"The dependence of the confidence interval\nvalue on the level of significance")

    plt.ylabel("Level of significance")
    plt.xlabel("Confidence interval length")

    plt.plot(x_deviation_lengths, quantiles.keys(), label=f"X component")
    plt.plot(y_deviation_lengths, quantiles.keys(), label=f"Y component")
    plt.legend()
    plt.show()
