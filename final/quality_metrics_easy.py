import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class PolynomialRegressionOptimizer:
    """
    Класс для оптимизации степени полиномиальной регрессии с использованием Grid Search.
    """

    def __init__(self, random_state: int = 42):
        """
        Инициализация оптимизатора.

        Args:
            random_state: Seed для воспроизводимости результатов
        """
        self.random_state = random_state
        self.best_degree = None
        self.best_model = None
        self.best_r2 = -np.inf
        self.results = {}
        self.X_test = None
        self.y_test = None

    def _create_polynomial_features(self, X: np.ndarray, degree: int) -> np.ndarray:
        """
        Создание полиномиальных признаков.

        Args:
            X: Исходные признаки
            degree: Степень полинома

        Returns:
            Массив с полиномиальными признаками
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X.reshape(-1, 1))

    def find_optimal_degree(self, X: np.ndarray, y: np.ndarray,
                            degrees_range: range = range(1, 11),
                            test_size: float = 0.2) -> Dict:
        """
        Поиск оптимальной степени полинома с использованием Grid Search.

        Args:
            X: Входные данные
            y: Целевая переменная
            degrees_range: Диапазон степеней для перебора
            test_size: Доля тестовой выборки

        Returns:
            Словарь с результатами поиска
        """
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        self.X_test = X_test
        self.y_test = y_test

        # Grid Search по степеням полинома
        for degree in degrees_range:
            try:
                # Создание полиномиальных признаков
                X_poly_train = self._create_polynomial_features(X_train, degree)

                # Обучение модели
                model = LinearRegression()
                model.fit(X_poly_train, y_train)

                # Оценка на тестовой выборке
                X_poly_test = self._create_polynomial_features(X_test, degree)
                y_pred = model.predict(X_poly_test)
                r2 = r2_score(y_test, y_pred)

                # Сохранение результатов
                self.results[degree] = {
                    'model': model,
                    'r2_score': r2,
                    'predictions': y_pred
                }

                # Обновление лучшей модели
                if r2 > self.best_r2:
                    self.best_r2 = r2
                    self.best_degree = degree
                    self.best_model = model

            except Exception as e:
                print(f"Ошибка при степени {degree}: {str(e)}")
                continue

        return self.results

    def get_best_model_info(self) -> Dict:
        """
        Получение информации о лучшей модели.

        Returns:
            Словарь с информацией о лучшей модели
        """
        if self.best_degree is None:
            raise ValueError("Оптимизация еще не выполнена. Вызовите метод find_optimal_degree()")

        return {
            'best_degree': self.best_degree,
            'best_r2_score': self.best_r2,
            'model': self.best_model
        }

    def predict_with_best_model(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание с использованием лучшей модели.

        Args:
            X: Входные данные для предсказания

        Returns:
            Предсказанные значения
        """
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала вызовите find_optimal_degree()")

        X_poly = self._create_polynomial_features(X, self.best_degree)
        return self.best_model.predict(X_poly)

    def visualize_results(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Визуализация результатов Grid Search и предсказаний лучшей модели.

        Args:
            X: Исходные данные
            y: Исходные целевые значения
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. График R2-score для разных степеней
        degrees = list(self.results.keys())
        r2_scores = [self.results[d]['r2_score'] for d in degrees]

        axes[0, 0].plot(degrees, r2_scores, 'o-', linewidth=2, markersize=8)
        axes[0, 0].axvline(x=self.best_degree, color='r', linestyle='--',
                           label=f'Лучшая степень: {self.best_degree}')
        axes[0, 0].set_xlabel('Степень полинома', fontsize=12)
        axes[0, 0].set_ylabel('R2-Score', fontsize=12)
        axes[0, 0].set_title('Grid Search: Зависимость R2-Score от степени полинома',
                             fontsize=14, pad=20)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 2. Предсказания лучшей модели
        X_sorted = np.sort(X)
        y_pred_best = self.predict_with_best_model(X_sorted)

        axes[0, 1].scatter(X, y, alpha=0.6, label='Исходные данные')
        axes[0, 1].plot(X_sorted, y_pred_best, 'r-', linewidth=3,
                        label=f'Полином {self.best_degree} степени\nR2 = {self.best_r2:.4f}')
        axes[0, 1].set_xlabel('X', fontsize=12)
        axes[0, 1].set_ylabel('y', fontsize=12)
        axes[0, 1].set_title('Предсказания лучшей модели', fontsize=14, pad=20)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Сравнение нескольких моделей
        sample_degrees = [1, self.best_degree, min(10, self.best_degree + 2)]
        for degree in sample_degrees:
            if degree in self.results:
                X_poly = self._create_polynomial_features(X_sorted, degree)
                y_pred = self.results[degree]['model'].predict(X_poly)
                axes[1, 0].plot(X_sorted, y_pred, '--', linewidth=2,
                                label=f'Степень {degree} (R2={self.results[degree]["r2_score"]:.4f})')

        axes[1, 0].scatter(X, y, alpha=0.3, s=20, label='Исходные данные')
        axes[1, 0].set_xlabel('X', fontsize=12)
        axes[1, 0].set_ylabel('y', fontsize=12)
        axes[1, 0].set_title('Сравнение моделей разной степени', fontsize=14, pad=20)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Остатки лучшей модели
        if self.X_test is not None:
            y_test_pred = self.predict_with_best_model(self.X_test)
            residuals = self.y_test - y_test_pred

            axes[1, 1].scatter(y_test_pred, residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Предсказанные значения', fontsize=12)
            axes[1, 1].set_ylabel('Остатки', fontsize=12)
            axes[1, 1].set_title('Диагностика модели: остатки vs предсказания',
                                 fontsize=14, pad=20)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Генерация синтетических данных для демонстрации
def generate_sample_data(n_samples: int = 100, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация синтетических данных для демонстрации работы.

    Args:
        n_samples: Количество образцов
        noise: Уровень шума

    Returns:
        Кортеж (X, y)
    """
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    # Используем полином 4-й степени для генерации данных
    y = (X**4 - 2*X**3 - 7*X**2 + 8*X + 12) + np.random.normal(0, noise, n_samples)
    return X, y


def main():
    """
    Основная функция демонстрации работы системы.
    """
    print("=" * 70)
    print("POLYNOMIAL REGRESSION GRID SEARCH OPTIMIZATION")
    print("=" * 70)

    # Генерация данных
    print("\n1. Генерация синтетических данных...")
    X, y = generate_sample_data()
    print(f"Сгенерировано {len(X)} образцов")

    # Инициализация и запуск оптимизатора
    print("\n2. Запуск Grid Search для степеней от 1 до 10...")
    optimizer = PolynomialRegressionOptimizer(random_state=42)
    results = optimizer.find_optimal_degree(X, y, degrees_range=range(1, 11))

    # Получение результатов
    best_info = optimizer.get_best_model_info()

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("=" * 70)
    print(f"Наилучшая степень полинома: {best_info['best_degree']}")
    print(f"R2-Score лучшей модели: {best_info['best_r2_score']:.4f}")

    # Детальная таблица результатов
    print("\nДетальные результаты Grid Search:")
    print("-" * 40)
    print(f"{'Степень':<10} {'R2-Score':<15}")
    print("-" * 40)
    for degree in sorted(results.keys()):
        print(f"{degree:<10} {results[degree]['r2_score']:<15.4f}")

    # Демонстрация предсказаний
    print("\n3. Демонстрация предсказаний лучшей модели...")
    sample_points = np.array([-2, 0, 2])
    predictions = optimizer.predict_with_best_model(sample_points)

    print("\nПример предсказаний:")
    for x, y_pred in zip(sample_points, predictions):
        print(f"   f({x:.1f}) ≈ {y_pred:.4f}")

    # Визуализация
    print("\n4. Генерация визуализаций...")
    optimizer.visualize_results(X, y)

    print("\n" + "=" * 70)
    print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()
