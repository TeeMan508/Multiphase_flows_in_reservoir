import numpy as np
import matplotlib.pyplot as plt
import math

K = 1
RO = 1
MU = 1
D_Z = 1

INJ_WELL_CORDS = [1/4, 1/4]
PROD_WELL_CORDS = [3/4, 3/4]

P_INJ_BOTTOM_HOLE = 1
P_PROD_BOTTOM_HOLE = - 1

WELL_RADIUS = 0.001


X = np.linspace(0, 1, 9)
Y = np.linspace(0, 1, 9)
h = (X[1] - X[0])

GAMMA = 2*math.pi * RO * K * D_Z / MU / np.log(WELL_RADIUS / (math.exp(-math.pi/2)*h))

k1 = 2
k2 = 6


def create_matrix():
    A = np.zeros((len(X) ** 2, len(X) ** 2))
    b = np.zeros((len(X)**2, 1))
    for i in range(0, len(X)**2):
        if i == k1*9 + k1:
            b[i] = P_INJ_BOTTOM_HOLE * GAMMA * h**2 / 4
        elif i == i == k2*9 + k2:
            b[i] = P_PROD_BOTTOM_HOLE * GAMMA * h**2 / 4
        for j in range(0, len(X)**2):
            boundary = False
            if i < 9:
                if j == i:
                    A[i, j] += 1
                elif j == i + 9:
                    A[i, j] += -1
                boundary = True
            if 72 <= i < 81:
                if j == i:
                    A[i, j] += 1
                elif j == i - 9:
                    A[i, j] += -1
                boundary = True
            if i % 9 == 0:
                if j == i:
                    A[i, j] += 1
                elif j == i + 1:
                    A[i, j] += -1
                boundary = True
            if i % 9 == 8:
                if j == i:
                    A[i, j] += 1
                elif j == i - 1:
                    A[i, j] = -1
                boundary = True

            if not boundary:
                if i == j == k1*9 + k1 or i == j == k2*9 + k2:
                    A[i, j] = (1 - GAMMA) * (h**2) / 4
                elif i == j:
                    A[i, j] = 1
                elif i == j + 9 or i == j - 9 or i == j - 1 or i == j + 1:
                    A[i, j] = -1 / 4

    return A, b


A, b = create_matrix()
result = np.linalg.solve(A, b)
result = result.reshape((9, 9))

plt.imshow(result, cmap='viridis', origin='lower', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)])
plt.colorbar(label='Colorbar Label')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('$P(x,y)$')

plt.show()
# plt.imshow(A, cmap='viridis')
# plt.colorbar()
# plt.show()