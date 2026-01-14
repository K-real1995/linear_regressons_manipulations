import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

x = np.linspace(1,10,num=10)
y = np.array(
    [1.,  3.,  4.,  2., 10.,  5.,  5.,  2.,  5., 10.],
    dtype=np.float32
)

# Создаём объект StandardScaler для нормализации данных
scaler = StandardScaler()

# Нормализуем y: сначала преобразуем в двумерный массив (требование fit_transform),
# применяем нормализацию (z-score: вычитаем среднее и делим на стандартное отклонение),
# затем flatten() возвращает одномерный массив для удобства
y_transformed = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Строим scatter-график: точки с координатами (x, y_transformed)
plt.scatter(x, y_transformed)

# Добавляем подпись оси X
plt.xlabel('x')

# Добавляем подпись оси Y (с указанием, что это z-score нормализованные y)
plt.ylabel('z-score y')

# Добавляем заголовок графика
plt.title('Нормализованные значения y (z-score) в зависимости от x')

# Включаем сетку на графике для лучшей читаемости
plt.grid(True)

#Визуализируем график
plt.show()