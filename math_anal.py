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






