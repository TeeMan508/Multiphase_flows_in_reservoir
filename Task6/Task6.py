# K_1=10
# K_2=100
# K_3=1
# K_4=10
#
# K = 4 * (K_1 + K_3) * (K_2 + K_4) * \
#     (K_2 * K_4 * (K_1 + K_3) + K_1 * K_3 * (K_2 + K_4)) * \
#     ((K_2 * K_4 * (K_1 + K_3) + K_1 * K_3 * (K_2 + K_4)) * (K_1 + K_2 + K_3 + K_4)
#      + 3 * (K_1 + K_2) *
#      (K_3 + K_4) * (K_1 + K_3) * (K_2 + K_4)) ** (-1)
#
# print(K)
import numpy as np
import matplotlib.pyplot as plt

phi_m = 0.2
phi_f = 0.01

mu = 1
bw = 1

A_c = 2 * 10**5
C_t = 10**(-6) * A_c
Lambda = 1

K_m = 0.01 * A_c
K_f = 50 * A_c

P_t0 = 1000
P_t_x0 = 500

T = 10
dt = 1

L = 10000
dx = 100
X = np.arange(0, L, dx)


def create_matrix(P_prev):
    A = np.zeros((2*len(X), 2*len(X)))
    b = np.zeros(2*len(X))
    for i in range(0, 2*len(X)):
        boundary = False
        if i == 0:
            b[i] = P_t_x0
        elif i == len(X) - 1:
            b[i] = 0
        elif i == len(X):
            b[i] = P_t_x0
        elif i == 2*len(X) - 1:
            b[i] = 0
        else:
            if i < len(X):
                b[i] = phi_m * C_t * P_prev[i] / dt
            else:
                b[i] = phi_f * C_t * P_prev[i] / dt
        for j in range(0, 2*len(X)):
            if i == j == 0:
                A[i, j] = 1
                boundary = True
            if i == j == (len(X) - 1):
                A[i, j] = 1
                A[i, j-1] = -1
                boundary = True
            if i == j == (len(X)):
                A[i, j] = 1
                boundary = True
            if i == j == (2*len(X) - 1):
                A[i, j] = 1
                A[i, j-1] = -1
                boundary = True

            if not boundary and i != 0 and i != 99 and i != 100 and i != 199:
                if i < len(X):
                    if i == j:
                        A[i, j] = phi_m * C_t / dt + 2 * K_m / mu / (dx**2) - Lambda

                    if i == j + 1:
                        A[i, j] = - K_m / mu / (dx**2)

                    if i == j - 1:
                        A[i, j] = - K_m / mu / (dx**2)

                    if i == j - len(X):
                        A[i, j] = Lambda

                else:
                    if i == j:
                        A[i, j] = phi_f * C_t / dt + 2 * K_f / mu / (dx**2) - Lambda

                    if i == j + 1:
                        A[i, j] = - K_f / mu / (dx**2)

                    if i == j - 1:
                        A[i, j] = - K_f / mu / (dx**2)

                    if i == j + len(X):
                        A[i, j] = Lambda
    # A[0, 0] = 1
    # A[len(X) - 1, len(X) - 1] = 1
    # A[len(X) - 1, len(X) - 2] = -1
    # A[len(X), len(X)] = 1
    # A[2 * len(X) - 1, 2 * len(X) - 1] = 1
    # A[2 * len(X) - 1, 2 * len(X) - 2] = 1
    return A, b


def solve():
    P0 = [float(P_t0*1.1) for i in range(0, len(X))] + [P_t0 for i in range(0, len(X))]
    # P0 = [float(P_t0) for i in range(0, 2*len(X))]
    plt.plot(X, P0[0:len(X)], label="P_m")
    plt.plot(X, P0[len(X):], label="P_f")
    plt.legend()
    plt.title(f"Time step: {0}")
    plt.grid()
    plt.show()

    for t in range(1, int(T/dt)+1):
        A, b = create_matrix(P0)
        P0 = np.linalg.solve(A, b)
        plt.plot(X, P0[0:len(X)], label="P_m")
        plt.plot(X, P0[len(X):], label="P_f")
        plt.legend()
        plt.title(f"Time step: {round(t*dt, 2)}")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # A, b = create_matrix([P_t0 for i in range(0, 2*len(X))])
    # plt.imshow(A, cmap='viridis')
    # plt.colorbar()
    # plt.show()
    # print(b)
    solve()
