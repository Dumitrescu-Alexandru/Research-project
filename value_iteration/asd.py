import numpy as np
import matplotlib.pyplot as plt

gamma = 0.995


# k = 0.01


def hyperbolic_coefs(k):
    no_value_fcns = 100
    coeffs = []
    gamma_intervals = np.linspace(0, 1, no_value_fcns)
    gamma_intervals = gamma_intervals
    for i in range(1, no_value_fcns - 1):
        coeffs.append(
            ((gamma_intervals[i + 1] - gamma_intervals[i]) * (1 / k) * gamma_intervals[i] ** (1 / k - 1)))
    coeffs = np.array(coeffs)
    return coeffs



def direct_computing(k):
    no_value_fcns = 100
    coeffs = []
    gamma_intervals = np.linspace(0, 1, no_value_fcns)
    gamma_intervals = gamma_intervals
    for i in range(1, no_value_fcns - 1):
        coeffs.append(
            -10.01*((gamma_intervals[i + 1] - gamma_intervals[i]) * (1 / k) * gamma_intervals[i] ** (1 / k - 1)))
    coeffs = np.array(coeffs)
    return coeffs / sum(coeffs)



def hyperbolic_fcn(value):
    coeffs = []
    gamma_intervals = np.linspace(0, 1, 50 + 1)
    gamma_intervals = gamma_intervals[1:]
    for i in range(49):
        coeffs.append((gamma_intervals[i + 1] - gamma_intervals[i]) * (1 / k) * gamma_intervals[i] ** (1 / (k - 1)))
    coeffs = np.array(coeffs)
    return sum(value * coeffs)


# t = np.array(list(range(1000)))
#
# gamma = 0.995
#
# plt.plot(t, 1 / (1 + k * t))
# plt.plot(t, gamma ** t)
# plt.legend(["Hyperbolic", "Exponential"])
# plt.show()
no_value_fcns = 100
k = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
print([sum(hyperbolic_coefs(k_)) for k_ in k])
print([sum(hyperbolic_coefs(k_) * np.array([-10.01] * (no_value_fcns-2))) for k_ in k])
print([-10.01 * 1 / (1 + k_) for k_ in k])
