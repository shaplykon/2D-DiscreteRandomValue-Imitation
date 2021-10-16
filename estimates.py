import math

import numpy as np
from scipy.optimize import fsolve

from constants import quantiles
from scipy.stats import chi2
from constants import THEORETICAL_MISES


def component_estimates(empirical_values, theoretical_values, p, component):
    deviation_lengths = []

    empirical_mean = np.sum(empirical_values) / len(empirical_values)
    theoretical_mean = np.sum(theoretical_values * p)

    empirical_variance = (np.sum(((empirical_values - empirical_mean) ** 2))) / len(empirical_values)
    theoretical_variance = np.sum((theoretical_values ** 2) * p) - theoretical_mean ** 2

    print(f'\n{component} component point estimates:')

    print(f'Theoretical (M({component})): {theoretical_mean}')
    print(f'Empirical (M({component})): {empirical_mean}')

    for quantile in quantiles.items():
        deviation = quantile[1] * math.sqrt(empirical_variance) / math.sqrt(len(empirical_values))
        if quantile[0] == "95":
            print(f'\nConfidence interval of M({component}) (significance level = 0.{quantile[0]})): '
                  f'({empirical_mean - deviation} : {empirical_mean + deviation}) ')
            print(f'Confidence interval length: {2 * deviation}')
        deviation_lengths.append(deviation)

    print(f'\nTheoretical (D({component})): {theoretical_variance}')
    print(f'Empirical (D({component})): {empirical_variance}')

    N = len(empirical_values)

    # Исправленное среднеквадратичное ожидание
    S = math.sqrt((np.sum(((empirical_values - empirical_mean) ** 2))) / N)

    # significance_level = 0.95
    # Критические точки хи-квадрат распределения
    chi_left = fsolve(lambda x: chi2.cdf(x, N - 1) - 0.05, N)[0]
    chi_right = fsolve(lambda x: chi2.cdf(x, N - 1) - (1 - 0.05), N)[0]

    left_border = (math.sqrt(len(empirical_values) - 1) * S) / math.sqrt(chi_right)
    right_border = (math.sqrt(len(empirical_values) - 1) * S) / math.sqrt(chi_left)

    print(
        f'Confidence interval of D({component}): '
        f'{left_border ** 2} : '
        f'{right_border ** 2}')

    print(f'\nTheoretical σ({component}): {math.sqrt(theoretical_variance)}')
    print(f'Empirical σ({component}): {math.sqrt(empirical_variance)}')

    print(
        f'Confidence interval of σ({component}): '
        f'{left_border} : '
        f'{right_border}')

    return deviation_lengths


def DRV_estimates(x_empirical, y_empirical, x_values, y_values, p):
    x_probabilities = np.sum(p, axis=1)
    y_probabilities = np.sum(p, axis=0)

    x_theoretical_mean = np.sum(x_values * x_probabilities)
    y_theoretical_mean = np.sum(y_values * y_probabilities)

    x_theoretical_variance = np.sum((x_values ** 2) * x_probabilities) - x_theoretical_mean ** 2
    y_theoretical_variance = np.sum((y_values ** 2) * y_probabilities) - y_theoretical_mean ** 2

    theoretical_covariation = 0
    length = len(p[0])

    for index in range(len(p.ravel()) - 1):
        i = index // length
        j = index % length
        theoretical_covariation += p[i][j] * (x_values[i] - x_theoretical_mean) * (y_values[j] - y_theoretical_mean)

    print(f'\nTheoretical covariation: {theoretical_covariation}')
    print(
        f'Theoretical correlation coefficient: {theoretical_covariation / math.sqrt(x_theoretical_variance * y_theoretical_variance)}')

    x_empirical_mean = np.sum(x_empirical) / len(x_empirical)
    y_empirical_mean = np.sum(y_empirical) / len(y_empirical)

    x_empirical_variance = (np.sum(((x_empirical - x_empirical_mean) ** 2))) / len(x_empirical)
    y_empirical_variance = (np.sum(((y_empirical - y_empirical_mean) ** 2))) / len(y_empirical)

    empirical_covariation = np.sum((x_empirical - x_empirical_mean) * (y_empirical - y_empirical_mean)) / (
        len(x_empirical))

    empirical_correlation_coefficient = empirical_covariation / (
        math.sqrt(x_empirical_variance * y_empirical_variance))

    print(f'\nEmpirical covariation: {empirical_covariation}')
    print(f'Empirical correlation coefficient: {empirical_correlation_coefficient}')

    # t - критерий Стьюдента(табличное значение) для a = 0.05
    # a - уровень значимости

    t = 1.96

    left_border = empirical_correlation_coefficient - t * (
            (1 - empirical_correlation_coefficient ** 2) / math.sqrt(len(x_empirical)))
    right_border = empirical_correlation_coefficient + t * (
            (1 - empirical_correlation_coefficient ** 2) / math.sqrt(len(x_empirical)))
    print(f'\nConfidence interval of correlation coefficient: '
          f'{left_border} : '
          f'{right_border}\n')


def mises_criteria(p_theoretical, p_empirical):
    m = len(p_theoretical)
    n = len(p_theoretical[0])
    empirical_mises_criteria = 0
    for i in range(m):
        for j in range(n):
            empirical_mises_criteria += (p_empirical[i][j] - p_theoretical[i][j]) ** 2
    empirical_mises_criteria += (1 / (12 * m * n))

    print(f'Empirical mises criteria: {empirical_mises_criteria}')
    print(f'Theoretical mises(confidence level 0.95): 0.461')

    if empirical_mises_criteria < THEORETICAL_MISES:
        print(f'No reason to reject H0 hypothesis({empirical_mises_criteria} < {THEORETICAL_MISES})')
    else:
        print(f'There is reason to reject H0 hypothesis ({empirical_mises_criteria} > {THEORETICAL_MISES})')
