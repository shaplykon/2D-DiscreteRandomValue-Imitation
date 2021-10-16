from estimates import *
from constants import *
from plots import *
from RandomVariableGenerator import *

if theoretical_probabilities.sum() != 1:
    raise Exception("Incorrect probability matrix definition!")

generator = RandomVariableGenerator(theoretical_X, theoretical_Y, theoretical_probabilities)

for _ in range(COUNT):
    k, r = generator.generate_value()
    empirical_probabilities[k][r] += 1
    empirical_X_values.append(theoretical_X[k])
    empirical_Y_values.append(theoretical_Y[r])

print("\nTheoretical probabilities matrix:\n", theoretical_probabilities)
print("\nEmpirical probabilities matrix:\n", empirical_probabilities / COUNT)

x_deviation_lengths = component_estimates(
    empirical_X_values, theoretical_X, np.sum(theoretical_probabilities, axis=1), 'X')

y_deviation_lengths = component_estimates(
    empirical_Y_values, theoretical_Y, np.sum(theoretical_probabilities, axis=0), 'Y')

DRV_estimates(empirical_X_values, empirical_Y_values, theoretical_X, theoretical_Y, theoretical_probabilities)

mises_criteria(empirical_probabilities / COUNT, theoretical_probabilities)

if input("Show histograms?(Y/N)").upper() == 'Y':
    show_histogram(empirical_X_values, "X")
    show_histogram(empirical_Y_values, "Y")
    show_common_histogram(empirical_X_values, empirical_Y_values)
    show_deviation_dependence_plot(x_deviation_lengths, y_deviation_lengths)
