import numpy as np
from scipy import integrate
import math
import matplotlib.pyplot as plt
h = 0.1

X = np.arange(0, 1+h, h)


def f(x):
    return 4 * (math.pi**2) * math.sin(2*math.pi*x)


def phi(i, x):
    if X[i-1] <= x <= X[i]:
        return (x - X[i-1]) / (X[i] - X[i-1])

    elif X[i] <= x <= X[i+1]:
        return (X[i+1] - x) / (X[i+1] - X[i])
    else:
        return 0


def d_phi(i, x):
    if X[i-1] <= x <= X[i]:
        return 1 / (X[i] - X[i-1])

    elif X[i] <= x <= X[i+1]:
        return -1 / (X[i+1] - X[i])
    else:
        return 0


def solve():
    A = np.eye(len(X))
    b = np.zeros((len(X), 1))
    for i in range(1, len(X)-1):
        print(f"step = {i}")
        def target_0(x):
            return phi(i, x)*f(x)

        b[i, 0] = integrate.quad(target_0, 0, 1, points=np.linspace(X[i-1], X[i+1], 3).tolist())[0]

        for j in range(1, len(X)-1):
            def target_1(x):
                return d_phi(j, x)*d_phi(i, x)

            A[i, j] = integrate.quad(target_1, 0, 1, points=np.linspace(X[i-1], X[i+1], 3).tolist())[0]

    alpha = np.linalg.solve(A, b)

    def p(x):
        s = 0
        for i in range(1, len(alpha)-1):
            s += alpha[i][0]*phi(i, x)
        return s
    return p


def calc_norm(res):
    reference = np.sin(2*math.pi*X)
    solution = np.array([res(x) for x in X])
    print(reference)
    print(solution)
    ab = np.abs((solution - reference))
    print(ab)
    ab = ab[~np.isnan(ab)]

    return np.max(ab)


if __name__ == "__main__":
    res = solve()
    c_norm = calc_norm(res)
    plt.plot(X, [res(x) for x in X], label="Solution")
    plt.plot(np.linspace(0,1,10000), [np.sin(2*math.pi*x) for x in np.linspace(0,1,10000)], label="Reference")
    plt.title(f"h={h}, norm={c_norm}")
    plt.legend()
    plt.grid()
    plt.show()
