import matplotlib.pyplot as plt
import numpy as np

def linear_regression_plot(x, y):
    """
    Функция вычисляет коэффициенты линейной регрессии w0 и w1,
    строит предсказания y_pred и визуализирует точки с линией регрессии.

    Параметры:
    x: список или массив значений x
    y: список или массив значений y (должен быть той же длины, что и x)
    """
    # Преобразуем в NumPy массивы для удобства
    x = np.array(x)
    y = np.array(y)

    n = len(x)  # Количество точек

    if n != len(y) or n < 2:
        raise ValueError("x и y должны иметь одинаковую длину и не менее 2 точек")

    # Вычисляем суммы
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    # Коэффициент w1 (наклон)
    w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

    # Коэффициент w0 (смещение)
    w0 = (sum_y - w1 * sum_x) / n

    # Предсказания y_pred
    y_pred = w0 + w1 * x

    # Визуализация
    plt.scatter(x, y, color='blue', label='Точки данных')
    plt.plot(x, y_pred, color='red', label='Линия регрессии')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Точки и линейная регрессия (w0={w0:.2f}, w1={w1:.4f})')
    plt.legend()
    plt.show()

    # Возвращаем коэффициенты и предсказания (опционально)
    return w0, w1, y_pred

# Пример вызова
x_hw = [50, 60, 70, 100]
y_hw = [10, 15, 40, 45]
w0, w1, y_pred = linear_regression_plot(x_hw, y_hw)
print(f"w0: {w0}, w1: {w1}")
print("Предсказания:", y_pred)