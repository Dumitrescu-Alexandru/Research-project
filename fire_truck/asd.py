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
            -10.01 * ((gamma_intervals[i + 1] - gamma_intervals[i]) * (1 / k) * gamma_intervals[i] ** (1 / k - 1)))
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


no_value_fcns = 1000
k = 0.3


def true_hyperbolic(t):
    return 1 / (1 + k * t)


def hyperbolic_coefs(t):
    coeffs = []
    gamma_intervals = np.linspace(0, 1, no_value_fcns)
    gamma_intervals = gamma_intervals
    for i in range(1, no_value_fcns - 1):
        coeffs.append(
            ((gamma_intervals[i + 1] - gamma_intervals[i]) * (1 / k) * gamma_intervals[i] ** ((1 / k) - 1)))
    coeffs = np.array(coeffs)
    gamma_intervals = gamma_intervals[1:no_value_fcns - 1]
    result = coeffs @ np.power(gamma_intervals, t)

    return result


T = np.array(range(100))
true_hyp = []
coef_hyp = []
for t_ in T:
    true_hyp.append(true_hyperbolic(t_))
    coef_hyp.append(hyperbolic_coefs(t_))

plt.plot(T, true_hyp, label="true")

plt.plot(T, coef_hyp, label="approx")
plt.legend()
plt.show()