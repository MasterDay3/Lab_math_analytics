from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

student_id = 43
assigned_function = student_id % 5 + 1
print(f"+++ Номер функції: {assigned_function} +++")




e = np.e
x = sp.Symbol('x')
func_for_taylor = sp.sqrt(x) * sp.exp(-x / 3)

function_expr = lambda x: np.sqrt(x) * np.exp(-x / 3)

# draw func
function = sp.sqrt(x) * sp.exp(-x / 3)
x_values = np.linspace(0, 20, 400)
y_values = function_expr(x_values)

plt.plot(x_values, y_values)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


def numeric_derivative(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    Обчислює похідну за стандартною формулою.
    :param func: Функція f(x), яку потрібно диференціювати.
    :param x: Точка, в якій обчислюється похідна.
    :param h: Розмір приросту (default: 1e-7).
    :return: Наближене значення f'(x).
    """
    derivative = (func(x + h) - func(x)) / h
    return derivative

def numeric_derivative_cd(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    Обчислює похідну за формулою центральної різниці.
    :param func: Функція f(x), яку потрібно диференціювати.
    :param x: Точка, в якій обчислюється похідна.
    :param h: Розмір приросту (default: 1e-7).
    :return: Наближене значення f'(x).
    """
    derivative_cd = (func(x + h) - func(x - h)) / (2*h)
    return derivative_cd

def numeric_second_derivative(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Обчислює другу похідну за формулою центральної різниці.
    :param func: Функція f(x), яку потрібно диференціювати.
    :param x: Точка, в якій обчислюється друга похідна.
    :param h: Розмір приросту (default: 1e-5).
    :return: Наближене значення f''(x).
    """
    g1 = (func(x + h) - func(x)) / h
    g2 = (func(x) - func(x - h)) / h
    second_derivative = (g1 - g2)/ h
    return second_derivative

def get_taylor_poly(x0: float, n: int):
    """
    Створює поліном Тейлора n-го порядку для f_sym в точці a.
    (Використовує глобальні символьні змінні `f_sym` та `x_sym`
    для розрахунку.)
    :param a: Точка, навколо якої будується розклад (point of expansion).
    :param n: Порядок полінома.
    :return: Числова функція, готова для обчислень з numpy-масивами.

    PS:
    Використовуйте sp.lambdify(x_sym, ..., 'numpy')` для
    перетворення символьного результату (який повертає sp.series)
    на числову функцію, яку можна передати в matplotlib для подальшої візуалізації!
    """
    taylor_row = func_for_taylor.series(x, x0, n+1).removeO()  # прибираємо O(...)
    taylor_func = sp.lambdify(x, taylor_row, 'numpy')
    return taylor_func
#print(get_taylor_poly(0.6, 3))

def bisection_method(func: Callable[[float], float], 
                       a: float, 
                       b: float, 
                       tol: float = 1e-8) -> float:
    """
    Знаходить корінь похідної f'(x) на інтервалі [a, b] методом бісекції.
    Це еквівалентно пошуку локального екстремуму оригінальної функції f(x).
    :param func: базова функція f(x), екструмум якої шукаємо
    :param a: Ліва межа інтервалу.
    :param b: Права межа інтервалу.
    :param tol: Точність (критерій зупинки, default: 1e-8).
    :return: Значення x, що є коренем f'(x) (точка екстремуму).
    """
    x = sp.Symbol('x')
    func_expr = func(x) if isinstance(func(x), sp.Basic) else sp.sympify(func(x))
    dfdx_expr = sp.diff(func_expr, x)
    dfdx = sp.lambdify(x, dfdx_expr, 'numpy')  # похідна для обчислень

    # Перевірка, що на межах знак змінюється
    if dfdx(a) * dfdx(b) > 0:
        return "Функція похідної має однаковий знак на кінцях інтервалу. Метод бісекції не спрацює."

    # Бісекція
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if dfdx(c) == 0:  # знайшли точний корінь
            return c
        elif dfdx(a) * dfdx(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def golden_section_search(func: Callable[[float], float], 
                          a: float, 
                          b: float, 
                          tol: float = 1e-8) -> float:
    """
    Знаходить мінімум унімодальної функції f(x) на інтервалі [a, b]
    методом золотого поділу.

    :param func: функція f(x), мінімум якої шукаємо
    :param a: ліва межа інтервалу
    :param b: права межа інтервалу
    :param tol: точність (критерій зупинки)
    :return: x, що мінімізує f(x) на [a, b]
    """
    gr = (np.sqrt(5) + 1) / 2 
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

def newtons_method(func: Callable[[float], float], 
                   x0: float, 
                   tol: float = 1e-8, 
                   max_iter: int = 100) -> float:
    """
    Знаходить корінь похідної f'(x) методом Ньютона (методом дотичних).
    Використовує ітеративну формулу: x_{n+1} = x_n - f'(x_n) / f''(x_n)
    
    :param func: базова функція f(x), екстремум якої шукаємо
    :param x0: Початкове наближення (точка)
    :param tol: Точність (критерій зупинки)
    :param max_iter: Максимальна кількість ітерацій
    :return: x, що є коренем f'(x) (точка екстремуму)
    """
    x = sp.Symbol('x')
    func_expr = func(x) if isinstance(func(x), sp.Basic) else sp.sympify(func(x))
    dfdx_expr = sp.diff(func_expr, x)
    d2fdx2_expr = sp.diff(dfdx_expr, x)    
    f_prime = sp.lambdify(x, dfdx_expr, 'numpy')
    f_double_prime = sp.lambdify(x, d2fdx2_expr, 'numpy')
    xn = x0
    for _ in range(max_iter):
        f1 = f_prime(xn)
        f2 = f_double_prime(xn)
        
        if f2 == 0:  # щоб уникнути ділення на нуль
            return "Друга похідна стала нулем, метод Ньютона не працює."
        
        xn_new = xn - f1 / f2
        
        if abs(xn_new - xn) < tol:
            return xn_new
        
        xn = xn_new
    
    # Якщо не збіглися за max_iter
    return 'Не збіглись за max_iter'







