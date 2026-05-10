import numpy as np
import matplotlib.pyplot as plt


# Функция для построения графиков
def graph(x, y1, label1, y2=None, label2=None, y_dot=None, label_dot=None,
          x_title=None, y_title=None, title=None, log=False):

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, 'b-', linewidth=1.5, label=label1)

    if (y2 is not None) and (label2 is not None):
        plt.plot(x, y2, 'g-', linewidth=1.5, label=label2)

    if (y_dot is not None) and (label_dot is not None):
        plt.plot(x, y_dot, 'r--', linewidth=1.5, label=label_dot)

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
def projection_graphs(coords_1, coords_2, proj, label1, label2):
    if len(proj) != 4 or len(proj[0]) != 3:
        raise ValueError('Неверный формат проекций')

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(np.real(coords_1[proj[0, 0] - 1]), np.imag(coords_1[proj[0, 0] - 1]), 'b-', linewidth=1.5,
             label=label1)
    ax1.plot(np.real(coords_2[proj[0, 0] - 1]), np.imag(coords_2[proj[0, 0] - 1]), 'r-', linewidth=1.5,
             label=label2)
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imag')
    ax1.set_title(f'Проекция на ось O_X{proj[0, 0]}')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(np.real(coords_1[proj[0, 1] - 1]), np.imag(coords_1[proj[0, 1] - 1]), 'b-', linewidth=1.5,
             label=label1)
    ax2.plot(np.real(coords_2[proj[0, 1] - 1]), np.imag(coords_2[proj[0, 1] - 1]), 'r-', linewidth=1.5,
             label=label2)
    ax2.set_xlabel('Real')
    ax2.set_ylabel('Imag')
    ax2.set_title(f'Проекция на ось O_X{proj[0, 1]}')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(np.real(coords_1[proj[0, 2] - 1]), np.imag(coords_1[proj[0, 2] - 1]),
             'b-', linewidth=1.5, label=label1)
    ax3.plot(np.real(coords_2[proj[0, 2] - 1]), np.imag(coords_2[proj[0, 2] - 1]),
             'r-', linewidth=1.5, label=label2)
    ax3.set_xlabel('Real')
    ax3.set_ylabel('Imag')
    ax3.set_title(f'Проекция на ось O_X{proj[0, 2]}')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(np.real(coords_1[proj[1, 0] - 1]), np.real(coords_1[proj[1, 1] - 1]),
             np.real(coords_1[proj[1, 2] - 1]),
             'b-', linewidth=1.5, label=label1)
    ax4.plot(np.real(coords_2[proj[1, 0] - 1]), np.real(coords_2[proj[1, 1] - 1]),
             np.real(coords_2[proj[1, 2] - 1]),
             'r-', linewidth=1.5, label=label2)
    ax4.set_xlabel(f'Ось O_X{proj[1, 0]}')
    ax4.set_ylabel(f'Ось O_X{proj[1, 1]}')
    ax4.set_zlabel(f'Ось O_X{proj[1, 2]}')
    ax4.set_title(f'Проекция на O_X{proj[1, 0]}_X{proj[1, 1]}_X{proj[1, 2]} (real)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.plot(np.real(coords_1[proj[2, 0] - 1]), np.real(coords_1[proj[2, 1] - 1]),
             np.real(coords_1[proj[2, 2] - 1]),
             'b-', linewidth=1.5, label=label1)
    ax5.plot(np.real(coords_2[proj[2, 0] - 1]), np.real(coords_2[proj[2, 1] - 1]),
             np.real(coords_2[proj[2, 2] - 1]),
             'r-', linewidth=1.5, label=label2)
    ax5.set_xlabel(f'Ось O_X{proj[2, 0]}')
    ax5.set_ylabel(f'Ось O_X{proj[2, 1]}')
    ax5.set_zlabel(f'Ось O_X{proj[2, 2]}')
    ax5.set_title(f'Проекция на O_X{proj[2, 0]}_X{proj[2, 1]}_X{proj[2, 2]} (real)')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.6)

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.plot(np.real(coords_1[proj[3, 0] - 1]), np.real(coords_1[proj[3, 1] - 1]),
             np.real(coords_1[proj[3, 2] - 1]),
             'b-', linewidth=1.5, label=label1)
    ax6.plot(np.real(coords_2[proj[3, 0] - 1]), np.real(coords_2[proj[3, 1] - 1]),
             np.real(coords_2[proj[3, 2] - 1]),
             'r-', linewidth=1.5, label=label2)
    ax6.set_xlabel(f'Ось O_X{proj[3, 0]}')
    ax6.set_ylabel(f'Ось O_X{proj[3, 1]}')
    ax6.set_zlabel(f'Ось O_X{proj[3, 2]}')
    ax6.set_title(f'Проекция на O_X{proj[3, 0]}_X{proj[3, 1]}_X{proj[3, 2]} (real)')
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

    for i in range(1, iteration + 1):
        v_rec = mu * (s_v - S @ v_rec) + v_rec
        errors.append(np.linalg.norm(v - v_rec))
        coords.append(v_rec)

        if errors[-1] < tol:
            limit = np.copy(i)
            break

    iter_list = np.arange(limit + 1)

    return (limit, v_rec, iter_list, np.array(errors) / np.linalg.norm(v),
            np.array(coords))


# Функция для полиномиального ускорения методом сопряженных градиентов
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

        if errors[-1] < tol:
            limit = np.copy(i)
            break

    iter_list = np.arange(limit + 1)

    return (limit, v_rec, iter_list, np.array(errors_s) / s_norm(v, S),
            np.array(errors) / np.linalg.norm(v), np.array(coords))


def conjugate_gradient_acceleration_another(v, c_coef, T, S, n, iteration=50, tol=1e-9):
    limit = iteration
    v_rec = np.zeros(n, dtype=complex)

    s_v = T @ c_coef
    r = np.copy(s_v)
    p = np.copy(s_v)
    lya = (r @ p.conj()) / (p @ (S @ p).conj())

    errors = [np.linalg.norm(v)]
    errors_s = [s_norm(v, S)]
    coords = [np.copy(v_rec)]

    for i in range(1, iteration + 1):
        s_p = S @ p
        v_rec += lya * p
        r -= lya * s_p

        p = r - (r @ s_p.conj()) / (p @ s_p.conj()) * p
        lya = (r @ p.conj()) / (p @ (S @ p).conj())

        errors.append(np.linalg.norm(v - v_rec))
        errors_s.append(s_norm(v - v_rec, S))
        coords.append(np.copy(v_rec))

        if errors[-1] < tol:
            limit = np.copy(i)
            break

    iter_list = np.arange(limit + 1)

    return (limit, v_rec, iter_list, np.array(errors_s) / s_norm(v, S),
            np.array(errors) / np.linalg.norm(v), np.array(coords))


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

    # Рассчет размерности фрейма (и доказательство задания №2.1)
    print(f'\nRank(T) = {np.linalg.matrix_rank(T)}\n')

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

    print('Вектор v можно разложить по фрейму: ')
    summand = [f"{np.round(g_coef[i], decimals=3)} * f_{i + 1}" for i in range(m)]

    lines = []
    for i in range(0, m, 3):
        lines.append(" + ".join(summand[i:(i + 3)]))

    text = f"v = {lines[0]} +\n"
    for line in lines[1:-1]:
        text += f"    {line} +\n"
    text += f"    {lines[-1]}"

    print(text)

    # Используем итерационные методы для восстановление вектора v
    lim_1, v_rec_1, x_1, y_1, coords_1 = classic_frame_algorithm(v, c_coef, T, S, n, A, B)
    lim_2, v_rec_2, x_2, y_2_s, y_2, coords_2 = conjugate_gradient_acceleration(v, c_coef, T, S, n)
    lim_3, v_rec_3, x_3, y_3_s, y_3, coords_3 = conjugate_gradient_acceleration_another(v, c_coef, T, S, n)

    sigma = (np.sqrt(B) - np.sqrt(A)) / (np.sqrt(A) + np.sqrt(B))
    err_est_1 = ((B - A) / (A + B)) ** x_1
    err_est_2_s = 2 * sigma ** x_2 / (1 + sigma ** (2 * x_2))

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

    print('\nВостановленный вектор v (другая версия метода сопряженных градиентов): ')
    print(np.round(v_rec_3, decimals=3))
    if lim_3 is not None:
        print(f'Достаточно {lim_3} итераций')

    # График зависимости ошибки приближения классического фреймового алгоритма от числа итераций
    graph(x=x_1, y1=(100 * y_1), label1='Ошибка классического метода',
          y_dot=(100 * err_est_1), label_dot='Оценка погрешности',
          x_title='Число итераций', y_title='Погрешность, %',
          title='Классический фреймовый алгоритм')

    # Графики зависимости ошибки приближения метода сопряженных градиентов от числа итераций
    graph(x=x_2, y1=(100 * y_2), label1='Ошибка метода сопряженных градиентов',
          y2=(100 * y_3[:len(y_2)]), label2='Ошибка метода сопряженных градиентов (другая версия)',
          x_title='Число итераций', y_title='Погрешность, %',
          title='Ошибка метода сопряженных градиентов')

    graph(x=x_2, y1=(100 * y_2_s), label1='Ошибка метода сопряженных градиентов',
          y2=(100 * y_3_s[:len(y_2)]), label2='Ошибка метода сопряженных градиентов (другая версия)',
          y_dot=(100 * err_est_2_s), label_dot='Оценка погрешности',
          x_title='Число итераций', y_title='Погрешность, %',
          title='Полиноминальное ускорение (S-норма)')

    # Логарифмические зависимости ошибок от числа итераций
    graph(x=x_1, y1=y_1, label1='Ошибка классического метода',
          y_dot=err_est_1, label_dot='Оценка погрешности',
          x_title='Число итераций', y_title='Погрешность',
          title='Ошибка классического метода (логарифмическая шкала)', log=True)

    graph(x=x_2, y1=y_2, label1='Ошибка метода сопряженных градиентов',
          y2=y_3[:len(y_2)], label2='Ошибка метода сопряженных градиентов (другая версия)',
          x_title='Число итераций', y_title='Погрешность',
          title='Ошибка метода сопряженных градиентов (логарифмическая шкала)', log=True)

    graph(x=x_2, y1=y_2_s, label1='Ошибка метода сопряженных градиентов',
          y2=y_3_s[:len(y_2)], label2='Ошибка метода сопряженных градиентов (другая версия)',
          y_dot=err_est_2_s, label_dot='Оценка погрешности',
          x_title='Число итераций', y_title='Погрешность, %',
          title='Ошибка метода сопряженных градиентов (логарифмическая шкала S-нормы)', log=True)

    # Демонстрация проекций на различные оси
    projections = np.array([[1, 16, 32], [1, 2, 3], [10, 15, 20], [30, 31, 32]])

    projection_graphs(coords_1.T, coords_2.T, projections,
                      label1='Классический фреймовый метод', label2='Метод сопряженных градиентов')
    projection_graphs(coords_2.T, coords_3.T, projections,
                      label1='Метод сопряженных градиентов', label2='Метод сопряженных градиентов (другая версия)')


if __name__ == "__main__":
    main()
