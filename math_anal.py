from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import time

student_id = 43
assigned_function = student_id % 5 + 1
print(f"+++ Номер функції: {assigned_function} +++")

e = np.e
x = sp.Symbol('x')
function = sp.sqrt(x) * sp.exp(-x / 3)
func_for_taylor = sp.sqrt(x) * sp.exp(-x / 3)
function_expr = lambda x: np.sqrt(x) * np.exp(-x / 3)
def draw_func():
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
    """
    derivative = (func(x + h) - func(x)) / h
    return derivative

def numeric_derivative_cd(func: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    Обчислює похідну за формулою центральної різниці.
    """
    derivative_cd = (func(x + h) - func(x - h)) / (2*h)
    return derivative_cd

def analyze_derivatives(function: sp.Expr,
                        x_symbol: sp.Symbol,
                        a: float,
                        b: float,
                        step: float = 0.1,
                        h: float = 1e-7) -> None:
    """
    Порівнює точність numeric_derivative і numeric_derivative_cd
    """
    f_num = sp.lambdify(x_symbol, function, "numpy")
    derivative_sym = sp.diff(function, x_symbol)
    f_derivative = sp.lambdify(x_symbol, derivative_sym, "numpy")

    x_range = np.arange(a, b + step, step)

    errors_default = []
    errors_central = []

    for val in x_range:
        if val - h < 0:
            continue
        true_val = float(f_derivative(val))
        approx_default = numeric_derivative(f_num, val, h)
        approx_central = numeric_derivative_cd(f_num, val, h)
        errors_default.append(abs(approx_default - true_val))
        errors_central.append(abs(approx_central - true_val))
    if not errors_default:
        return
    max_error_default = max(errors_default)
    max_error_central = max(errors_central)
    print("Максимальна похибка (за означенням похідної):",
          f"{max_error_default:.3}")
    print("Максимальна похибка (за формулою центральної різниці):",
          f"{max_error_central:.3}")

    if max_error_default > max_error_central:
        print("Краще використовувати - формулу центральної різниці.")
    elif max_error_default < max_error_central:
        print("Краще використовувати - формулу означення похідної.")
    else:
        print("За двома формулами однакова похибка, отже можна використовувати, що ту, що іншу.")

def numeric_second_derivative(func: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """
    Обчислює другу похідну за формулою центральної різниці.
    """
    g1 = (func(x + h) - func(x)) / h
    g2 = (func(x) - func(x - h)) / h
    second_derivative = (g1 - g2)/ h
    return second_derivative

def get_taylor_poly(x0: float, n: int):
    """
    Створює поліном Тейлора n-го порядку.
    """
    taylor_row = func_for_taylor.series(x, x0, n+1).removeO()
    taylor_func = sp.lambdify(x, taylor_row, 'numpy')
    return taylor_func

def taylor_poly_visualisation(poly):
    '''
    Візуалізує введений поліном Тейлора
    '''
    x_values = np.linspace(0, 20, 400)
    y_values = poly(x_values)

    plt.plot(x_values, y_values, label='Поліном Тейлора')
    plt.grid(True)
    plt.title('Візуалізація полінома Тейлора')
    plt.show()

def bisection_method(func: Callable[[float], float],
                       a: float,
                       b: float,
                       tol: float = 1e-8) -> float:
    """
    Знаходить корінь похідної f'(x).
    """

    x_sym = sp.Symbol('x')
    dfdx_expr = sp.diff(function, x_sym)
    dfdx = sp.lambdify(x_sym, dfdx_expr, 'numpy')

    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if abs(dfdx(c)) < tol:
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
    Знаходить МАКСИМУМ функції f(x).
    """
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if func(c) > func(d):
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
    Знаходить корінь похідної f'(x) методом Ньютона.
    """
    x_sym = sp.Symbol('x')
    dfdx_expr = sp.diff(function, x_sym)
    d2fdx2_expr = sp.diff(dfdx_expr, x_sym)
    f_prime = sp.lambdify(x_sym, dfdx_expr, 'numpy')
    f_double_prime = sp.lambdify(x_sym, d2fdx2_expr, 'numpy')
    xn = x0
    for _ in range(max_iter):
        f1 = f_prime(xn)
        f2 = f_double_prime(xn)
        # Без перевірки на нуль
        xn_new = xn - f1 / f2
        if abs(xn_new - xn) < tol:
            return xn_new
        xn = xn_new
    return xn

def compare_optimization_methods():
    """
    Порівнює методи бісекції, золотого перетину та Ньютона.
    """
    # тестові параметри для перевірки функцій
    true_extremum = 1.5
    a, b = 0.1, 5.0
    x0 = 2.0
    tol = 1e-8
    t0 = time.perf_counter()
    res_bi = bisection_method(function_expr, a, b, tol)
    t_bi = time.perf_counter() - t0
    err_bi = abs(res_bi - true_extremum)
    t0 = time.perf_counter()
    res_gold = golden_section_search(function_expr, a, b, tol)
    t_gold = time.perf_counter() - t0
    err_gold = abs(res_gold - true_extremum)
    t0 = time.perf_counter()
    res_new = newtons_method(function_expr, x0, tol)
    t_new = time.perf_counter() - t0
    err_new = abs(res_new - true_extremum)
    print("Метод                 Результат     Похибка   Час виконання (s)")
    print(f"{'Бісекція':<20} {res_bi:<12.8f} {err_bi:<12.2e} {t_bi:.6f}")
    print(f"{'Золотий перетин':<20} {res_gold:<12.8f}  {err_gold:.2e}  {t_gold:.6f}")
    print(f"{'Ньютон':<20} {res_new:<12.8f} {err_new:<12.2e} {t_new:.6f}")


#compare_optimization_methods()
#draw_func()
#print(numeric_derivative(lambda t: t**2, 3.0))
#print(numeric_derivative_cd(lambda t: t**2, 3.0))
# x = sp.Symbol('x')
# f = x**2
# print(analyze_derivatives(f, x, 0.0, 1.0, step=0.5, h=1e-5))
#print(numeric_second_derivative(lambda t: t**3, 1.0))
# f = lambda x: -(x - 1)**2 + 5   # максимум у точці x = 1
# print(golden_section_search(f, -2.0, 4.0, tol=1e-6))
# x_opt = newtons_method(function_expr, 2.0, tol=1e-8, max_iter=50)
# print(x_opt)
#compare_optimization_methods()
#print(taylor_poly_visualisation(get_taylor_poly(0.6, 3)))
#print(get_taylor_poly(0.6, 3))

