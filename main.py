import numpy as np
import matplotlib.pyplot as plt


# Функция для построения графиков
def graph(x1, y1, label1, x2=None, y2=None, label2=None, x_title=None, y_title=None, title=None, dotted=False, log=False):
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, 'b-', linewidth=1.5, label=label1)

    if (y2 is not None) and (label2 is not None):
        if x2 is not None:
            if dotted:
                plt.plot(x2, y2, 'r--', linewidth=1.5, label=label2)
            else:
                plt.plot(x2, y2, 'r-', linewidth=1.5, label=label2)
        else:
            if dotted:
                plt.plot(x1, y2, 'r--', linewidth=1.5, label=label2)
            else:
                plt.plot(x1, y2, 'r-', linewidth=1.5, label=label2)

    if log:
        plt.yscale('log')
    if label1 != '':
        plt.legend()
    if x_title:
        plt.xlabel(x_title)
    if y_title:
        plt.ylabel(y_title)
    if title:
        plt.title(title, fontsize=12, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# Функция для построения проекций
def projection_graphs(coords_1, coords_2):
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(np.real(coords_1[0]), np.imag(coords_1[0]), 'b-', linewidth=1.5,
             label='Классический фреймовый метод')
    ax1.plot(np.real(coords_2[0]), np.imag(coords_2[0]), 'r-', linewidth=1.5,
             label='Метод сопряженных градиентов')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imag')
    ax1.set_title('Проекция на ось O_X1')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(np.real(coords_1[15]), np.imag(coords_1[15]), 'b-', linewidth=1.5,
             label='Классический фреймовый метод')
    ax2.plot(np.real(coords_2[15]), np.imag(coords_2[15]), 'r-', linewidth=1.5,
             label='Метод сопряженных градиентов')
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imag')
    ax2.set_title('Проекция на ось O_X16')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(np.real(coords_1[31]), np.imag(coords_1[31]), 'b-', linewidth=1.5,
             label='Классический фреймовый метод')
    ax3.plot(np.real(coords_2[31]), np.imag(coords_2[31]), 'r-', linewidth=1.5,
             label='Метод сопряженных градиентов')
    ax3.set_xlabel('Real')
    ax3.set_ylabel('Imag')
    ax3.set_title('Проекция на ось O_X32')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(np.real(coords_1[0]), np.real(coords_1[1]), np.real(coords_1[2]),
             'b-', linewidth=1.5, label='Классический фреймовый метод')
    ax4.plot(np.real(coords_2[0]), np.real(coords_2[1]), np.real(coords_2[2]),
             'r-', linewidth=1.5, label='Метод сопряженных градиентов')
    ax4.set_xlabel('Ось O_X1')
    ax4.set_ylabel('Ось O_X2')
    ax4.set_zlabel('Ось O_X3')
    ax4.set_title('Проекция на O_X1_X2_X3 (real)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.plot(np.real(coords_1[9]), np.real(coords_1[14]), np.real(coords_1[19]),
             'b-', linewidth=1.5, label='Классический фреймовый метод')
    ax5.plot(np.real(coords_2[9]), np.real(coords_2[14]), np.real(coords_2[19]),
             'r-', linewidth=1.5, label='Метод сопряженных градиентов')
    ax5.set_xlabel('Ось O_X10')
    ax5.set_ylabel('Ось O_X15')
    ax5.set_zlabel('Ось O_X20')
    ax5.set_title('Проекция на O_X10_X15_X20 (real)')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.6)

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.plot(np.real(coords_1[29]), np.real(coords_1[30]), np.real(coords_1[31]),
             'b-', linewidth=1.5, label='Классический фреймовый метод')
    ax6.plot(np.real(coords_2[29]), np.real(coords_2[30]), np.real(coords_2[31]),
             'r-', linewidth=1.5, label='Метод сопряженных градиентов')
    ax6.set_xlabel('Ось O_X30')
    ax6.set_ylabel('Ось O_X31')
    ax6.set_zlabel('Ось O_X32')
    ax6.set_title('Проекция на O_X30_X31_X32 (real)')
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Сравнение фреймовых алгоритмов', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Функция для рассчета S-нормы вектора
def s_norm(x, S):
    return np.sqrt(np.abs(x @ (S @ x).conj()))


# Функция для классического фреймового алгоритма
def classic_frame_algorithm(v, c_coef, T, S, n, A, B, iteration=5000, tol=1e-9):
    limit = iteration
    v_rec = np.zeros(n, dtype=complex)

    s_v = T @ c_coef
    mu = 2 / (A + B)

    errors = [np.linalg.norm(v)]
    coords = [v_rec]
    angle = []

    for i in range(1, iteration + 1):
        v_rec = mu * (s_v - S @ v_rec) + v_rec
        errors.append(np.linalg.norm(v - v_rec))
        coords.append(v_rec)

        angle.append(np.arccos(
            np.clip(np.abs(v @ v_rec.conj()) / (np.linalg.norm(v) * np.linalg.norm(v_rec)), a_min=-1.0, a_max=1.0)
        ))

        if errors[-1] < tol:
            limit = np.copy(i)
            break

    iter_list = np.arange(limit + 1)

    return (limit, v_rec, iter_list, np.array(errors) / np.linalg.norm(v),
            np.array(coords).T, np.array(angle))


# Функция для полиномиального ускорения сопряженный градиентов
def conjugate_gradient_acceleration(v, c_coef, T, S, n, iteration=50, tol=1e-9):
    limit = iteration
    p_prev = np.zeros(n, dtype=complex)
    v_rec = np.zeros(n, dtype=complex)

    s_v = T @ c_coef
    r = np.copy(s_v)
    p = np.copy(s_v)
    lya = (r @ p.conj()) / (p @ (S @ p).conj())

    errors = [np.linalg.norm(v)]
    errors_s = [s_norm(v, S)]
    coords = [np.copy(v_rec)]
    angle = []

    for i in range(1, iteration + 1):
        s_p = S @ p
        s_p_prev = S @ p_prev
        v_rec += lya * p
        r -= lya * s_p

        if i == 1:
            p_new = s_p - (s_p @ s_p.conj()) / (p @ s_p.conj()) * p

        else:
            p_new = (s_p - (s_p @ s_p.conj()) / (p @ s_p.conj()) * p -
                     (s_p @ s_p_prev.conj()) / (p_prev @ s_p_prev.conj()) * p_prev)

        p_prev = np.copy(p)
        p = np.copy(p_new)
        lya = (r @ p.conj()) / (p @ (S @ p).conj())

        errors.append(np.linalg.norm(v - v_rec))
        errors_s.append(s_norm(v - v_rec, S))
        coords.append(np.copy(v_rec))

        angle.append(np.arccos(
            np.clip(np.abs(v @ v_rec.conj()) / (np.linalg.norm(v) * np.linalg.norm(v_rec)), a_min=-1.0, a_max=1.0)
        ))

        if errors[-1] < tol:
            limit = np.copy(i)
            break

    iter_list = np.arange(limit + 1)

    return (limit, v_rec, iter_list, np.array(errors_s) / s_norm(v, S), np.array(errors) / np.linalg.norm(v),
            np.array(coords).T, np.array(angle))


def main():
    # m - число n-мерных векторов, образующих фрейм
    m = 50
    n = 32

    # T - оператор синтеза (предфреймовый оператор), столбцы в котором соответствуют векторам f_k
    T = np.zeros((n, m), dtype=complex)

    for k in range(1, m + 1):
        j = np.arange(n)
        f_k = (k / np.sqrt(m)) * np.exp(2j * np.pi * j * (k - 1) / m)
        T[..., k - 1] = f_k

    # v - вектор [1, 2, ..., n]
    v = np.arange(1, n + 1, dtype=float)

    # Рассчет размерности фрейма (и доказательство)
    print(f'Rank(T) = {np.linalg.matrix_rank(T)}\n')

    # - фреймовый оператор
    S = T @ T.conj().T

    # eigenvalues - собственные значения фреймового оператора S
    eigenvalues = np.linalg.eigvalsh(S)

    # A и B - границы фрейма, соответствующие наименьшему и наибольшему собственному значению
    A = eigenvalues[0]
    B = eigenvalues[-1]
    print(f'A = {round(A, 3)}')
    print(f'B = {round(B, 3)}\n')

    # Набор скалярных произведений векторов f_k и вектора v
    c_coef = v @ T.conj()

    # Набор скалярных произведений векторов канонически двойственного фрейма и вектора v
    g_coef = v @ (np.linalg.inv(S) @ T).conj()
    print('Коэффициенты разложения вектора v: ')
    print(np.round(g_coef, decimals=3))

    # Используем итерационные методы для восстановление вектора v
    lim_1, v_rec_1, x_1, y_1, coords_1, angle_1 = classic_frame_algorithm(v, c_coef, T, S, n, A, B)
    lim_2, v_rec_2, x_2, y_2_s, y_2, coords_2, angle_2 = conjugate_gradient_acceleration(v, c_coef, T, S, n)

    sigma = (np.sqrt(B) - np.sqrt(A)) / (np.sqrt(A) + np.sqrt(B))
    err_basic_1 = ((B - A) / (A + B)) ** x_1
    err_basic_2_s = 2 * sigma ** x_2 / (1 + sigma ** (2 * x_2))

    print('\nИзначальный вектор v: ')
    print(v)

    print('\nВостановленный вектор v (через обратную матрицу S): ')
    print(np.round(T @ g_coef, decimals=3))

    print('\nВостановленный вектор v (классический фреймовый алгоритм): ')
    print(np.round(v_rec_1, decimals=3))
    if lim_1 is not None:
        print(f'Достаточно {lim_1} итераций')

    print('\nВостановленный вектор v (метод сопряженных градиентов): ')
    print(np.round(v_rec_2, decimals=3))
    if lim_2 is not None:
        print(f'Достаточно {lim_2} итераций')

    # График зависимости ошибки приближения классического фреймового алгоритма от числа итераций
    graph(x1=x_1, y1=(100 * y_1), label1='Ошибка классического метода',
          y2=(100 * err_basic_1), label2='Оценка погрешности',
          x_title='Число итераций', y_title='Погрешность, %',
          title='Классический фреймовый алгоритм', dotted=True)

    # Графики зависимости ошибки приближения метода сопряженных градиентов от числа итераций
    graph(x1=x_2, y1=(100 * y_2), label1='',
          x_title='Число итераций', y_title='Погрешность, %',
          title='Ошибка метода сопряженных градиентов')

    graph(x1=x_2, y1=(100 * y_2_s), label1='Ошибка метода сопряженных градиентов',
          y2=(100 * err_basic_2_s), label2='Оценка погрешности',
          x_title='Число итераций', y_title='Погрешность, %',
          title='Полиноминальное ускорение (S-норма)', dotted=True)

    # Логарифмические зависимости ошибок от числа итераций
    graph(x1=x_1, y1=y_1, label1='',
          x_title='Число итераций', y_title='Погрешность',
          title='Ошибка классического метода', log=True)

    graph(x1=x_2, y1=y_2, label1='',
          x_title='Число итераций', y_title='Погрешность',
          title='Ошибка метода сопряженных градиентов', log=True)

    # Демонстрация проекций на различные оси
    projection_graphs(coords_1, coords_2)

    # Графики зависимости угла расхождения от числа итераций
    graph(x1=x_1[1:], y1=(180.0 / np.pi * angle_1), label1='',
          x_title='Число итераций', y_title='Угол расхождения с вектором v',
          title='Расхождение векторов в классическом методе')

    graph(x1=x_2[1:], y1=(180.0 / np.pi * angle_2), label1='',
          x_title='Число итераций', y_title='Угол расхождения с вектором v',
          title='Расхождение векторов в методе сопряженных градиентов')


if __name__ == "__main__":
    main()
