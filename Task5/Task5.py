import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

dt = 0.1
T = 10
dx = 1
L = 100

Swr = 0.2
Sor = 0.15
Sw_t_x0 = 1 - Sor
Sw_t_xl = Swr

K_rw_star = 0.6
K_ro_star = 1
Nw = 2
No = 2


S_ali = 10 / 10  # %
T0 = 70  # grad C
T_k = (T0 + 273.15)  # grad K
T_f = T0*1.8 + 32
P0 = 187 * 14.6959  # psi

F_sv = 1 - 1.87 * (10**(-3)) * (S_ali**(1/2)) + 2.18 * (10**(-4)) * (S_ali**(5/2)) + \
       (T_f**(1/2) - 0.0135 * T_f) * (2.76 * (10**(-3)) * S_ali - 3.44 * (10**(-4)) * (S_ali**(3/2)))
F_pv = 1 + 3.5 * 10**(-12) * P0 * (T_f-40)

mu_w = 0.02414 * np.power(10, 247.8/(T_k-140)) * F_sv * F_pv

mu_o = 50  # cP

A = 1  # ft_3
q = 1  # ft/day
phi = 0.2
GAMMA = q / A / phi


def Swn(Sw):
    return (Sw - Swr) / (1 - Swr - Sor)


def Kro(Sw):
    if abs(Sw - Swr) < 0.0001:
        return K_ro_star
    return K_ro_star * ((1 - Swn(Sw))**No)


def Krw(Sw):
    if abs(Sw - 1 + Sor) < 0.0001:
        return K_rw_star
    return K_rw_star * Swn(Sw)**Nw


def lam_w(Sw):
    return Krw(Sw) / mu_w


def lam_o(Sw):
    return Kro(Sw) / mu_o


def fw(Sw):
    if Sw <= Swr:
        return 0
    elif Sw >= (1 - Swr):
        return 1
    return lam_w(Sw) / (lam_w(Sw) + lam_o(Sw))


def solve_semi_analytic(t):
    def target(Sw):
        return (fw(Sw + 0.01) - fw(Sw)) / 0.01 - fw(Sw) / (Sw - Swr)

    Swf = fsolve(target, 0.3)[0]

    Sw_list = np.linspace(Swr, 1 - Sor, 101)
    #print(Sw_list)

    def result(Sw):
        if Sw > Swf:
            # if abs(t - 1.) < 0.0001 and abs(Sw - 0.32) < 0.001:
                # print(fw(Sw), fw(Sw))
            return t * GAMMA * (fw(Sw + 0.01) - fw(Sw)) / 0.01
        else:
            return t * GAMMA * (fw(Swf + 0.01) - fw(Swf)) / 0.01

    res_x = [Sw_list.tolist(), []]
    for sw in Sw_list:
        res_x[1].append(result(sw))

    while res_x[1][0] < L:
        #print(res_x[1][0])
        #print(res_x[1][0])
        res_x[1].insert(0, res_x[1][0]+1)
        res_x[0].insert(0, Swr)
    return res_x


def solve():
    Sw_prev = [Swr for i in range(0, int(L / dx))]
    plt.plot([dx*i for i in range(0, int(L / dx))], Sw_prev)
    plt.savefig(f"./result/frame_{0}.png")
    plt.clf()
    #plt.show()
    for t in range(1, int(T / dt)):
        def F(Sw_vec):
            functions = [Sw_vec[0] - Sw_t_x0]
            for i in range(1, len(Sw_vec)-1):
                buf = Sw_vec[i] - Sw_prev[i] + GAMMA * dt * (fw(Sw_vec[i]) - fw(Sw_vec[i-1])) / dx
                functions.append(buf)
            functions.append(Sw_vec[len(Sw_vec)-1] - Sw_t_xl)
            return functions
        reference = solve_semi_analytic(t*dt)
        Sw_prev = fsolve(F, Sw_prev)
        # Sw_prev[0] = Sw_t_x0
        # Sw_prev[-1] = Sw_t_xl
        plt.plot([dx * i for i in range(0, int(L / dx))], Sw_prev, label="solution")
        plt.plot(reference[1], reference[0], label="Reference")
        #print(reference[1])
        plt.legend()
        plt.title(f"Time step: {round(t*dt, 1)}")
        plt.grid()
        plt.savefig(f"./result/frame_{t}.png")
        #plt.clf()
        plt.show()


def build_gif():
    frames = []
    for frame_number in range(1, 99):
        # Открываем изображение каждого кадра.
        frame = Image.open(f'./result/frame_{frame_number}.png')
        # Добавляем кадр в список с кадрами.
        frames.append(frame)

    # Берем первый кадр и в него добавляем оставшееся кадры.
    frames[0].save(
        './result/result.gif',
        save_all=True,
        append_images=frames[0:],  # Срез который игнорирует первый кадр.
        optimize=True,
        duration=100,
        loop=0
    )


if __name__ == "__main__":
    # sw = np.linspace(Swr, 1-Sor, 100)
    # k_rw_vec = [Krw(sw_i) for sw_i in sw]
    # k_ro_vec = [Kro(sw_i) for sw_i in sw]
    # plt.plot([Swn(sw_i) for sw_i in sw], k_rw_vec)
    # plt.plot([Swn(sw_i) for sw_i in sw], k_ro_vec)
    # plt.show()
    # print()
    # solve()
    # plt.plot([0.01*i for i in range(0, 100)], [fw(0.01*i) for i in range(0, 100)])
    # plt.grid()
    # plt.show()
    # print(mu_w)
    solve()
    build_gif()
    # solve()
    # Sw_list = np.linspace(Swr, 1-Sor, 100)
    # analytic, full_result = solve_semi_analytic()
    # #plt.plot([analytic(sw) for sw in Sw_list], Sw_list)
    # plt.plot(full_result[1], full_result[0], label="Reference")
    # plt.plot(res_t1[0], res_t1[1], label="Solution")
    # plt.title(f"Time step: {1}")
    # plt.grid()
    # plt.show()


