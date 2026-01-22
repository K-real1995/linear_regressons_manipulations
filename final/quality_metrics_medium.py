import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
import warnings

class CustomLinearReg:
    """
    Линейная регрессия с аналитическим вычислением коэффициентов через нормальное уравнение.
    
    Attributes
    ----------
    coef_ : np.ndarray
        Коэффициенты модели (веса признаков)
    intercept_ : float
        Свободный член (bias term)
    n_features_in_ : int
        Количество признаков во время обучения
    feature_names_in_ : Optional[np.ndarray]
        Имена признаков, если были переданы при обучении
    _is_fitted : bool
        Флаг, указывающий, была ли модель обучена
    singular_values_ : Optional[np.ndarray]
        Сингулярные значения матрицы признаков (для диагностики)
    condition_number_ : Optional[float]
        Число обусловленности матрицы X.T @ X
    """

    def __init__(self, fit_intercept: bool = True):
        """
        Инициализация модели линейной регрессии.
        
        Parameters
        ----------
        fit_intercept : bool, default=True
            Если True, добавляет свободный член к модели.
            Если False, модель проходит через начало координат.
        """
        self.fit_intercept = fit_intercept

        # Инициализация атрибутов модели
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._is_fitted = False
        self.singular_values_ = None
        self.condition_number_ = None

    def _validate_input(
            self,
            X: ArrayLike,
            y: Optional[ArrayLike] = None,
            reset: bool = False
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Валидация и преобразование входных данных.
        
        Parameters
        ----------
        X : ArrayLike
            Матрица признаков
        y : Optional[ArrayLike]
            Вектор целевой переменной
        reset : bool
            Сбрасывать ли информацию о признаках
            
        Returns
        -------
        tuple
            Валидированные X и y как numpy массивы
        """
        # Преобразование X в numpy array
        X = np.asarray(X, dtype=np.float64)

        # Проверка размерности X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"X должен быть 2D массивом, получено {X.ndim}D")

        # Сохранение информации о признаках при первом вызове
        if reset or not self._is_fitted:
            self.n_features_in_ = X.shape[1]

        # Проверка согласованности размеров
        if y is not None:
            y = np.asarray(y, dtype=np.float64).ravel()
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Несовпадение размеров: X имеет {X.shape[0]} образцов, "
                    f"y имеет {y.shape[0]} образцов"
                )

        return X, y

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Добавление столбца единиц для свободного члена.
        
        Parameters
        ----------
        X : np.ndarray
            Исходная матрица признаков
            
        Returns
        -------
        np.ndarray
            Матрица признаков с добавленным столбцом единиц
        """
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            sample_weight: Optional[ArrayLike] = None
    ) -> 'CustomLinearReg':
        """
        Обучение модели линейной регрессии.
        
        Parameters
        ----------
        X : ArrayLike формы (n_samples, n_features)
            Матрица признаков для обучения
        y : ArrayLike формы (n_samples,)
            Целевые значения
        sample_weight : Optional[ArrayLike] формы (n_samples,)
            Веса образцов. Если None, все образцы имеют равный вес.
            
        Returns
        -------
        self : CustomLinearReg
            Обученная модель
        """
        # Валидация входных данных
        X, y = self._validate_input(X, y, reset=True)

        # Проверка на достаточное количество образцов
        n_samples = X.shape[0]
        if n_samples < self.n_features_in_ + int(self.fit_intercept):
            warnings.warn(
                f"Количество образцов ({n_samples}) меньше чем "
                f"количество признаков ({self.n_features_in_}) + intercept. "
                "Это может привести к переобучению.",
                UserWarning
            )

        # Добавление intercept если требуется
        if self.fit_intercept:
            X_design = self._add_intercept(X)
        else:
            X_design = X

        # Применение весов образцов если предоставлены
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if len(sample_weight) != n_samples:
                raise ValueError("sample_weight должен иметь ту же длину, что и y")
            # Диагональная матрица весов
            W_sqrt = np.diag(np.sqrt(sample_weight))
            X_design = W_sqrt @ X_design
            y = W_sqrt @ y

        # Вычисление сингулярных значений для диагностики
        U, S, Vt = np.linalg.svd(X_design, full_matrices=False)
        self.singular_values_ = S
        self.condition_number_ = S[0] / S[-1] if S[-1] > 0 else np.inf

        # Предупреждение о мультиколлинеарности
        if self.condition_number_ > 1e12:
            warnings.warn(
                f"Сильная мультиколлинеарность обнаружена. "
                f"Число обусловленности: {self.condition_number_:.2e}",
                UserWarning
            )

        try:
            # Аналитическое решение через нормальное уравнение
            # θ = (X^T X)^(-1) X^T y
            # Используем псевдообратную матрицу для численной устойчивости
            theta = np.linalg.pinv(X_design) @ y
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "Матрица признаков сингулярна или плохо обусловлена. "
                "Попробуйте удалить коррелированные признаки или использовать регуляризацию."
            ) from e

        # Разделение intercept и коэффициентов
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        # Сохранение информации о признаках
        self._is_fitted = True

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Предсказание целевой переменной.
        
        Parameters
        ----------
        X : ArrayLike формы (n_samples, n_features)
            Матрица признаков для предсказания
            
        Returns
        -------
        y_pred : np.ndarray формы (n_samples,)
            Предсказанные значения
        """
        if not self._is_fitted:
            raise RuntimeError("Модель должна быть обучена перед вызовом predict")

        # Валидация входных данных
        X, _ = self._validate_input(X)

        # Проверка количества признаков
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Ожидалось {self.n_features_in_} признаков, "
                f"получено {X.shape[1]}"
            )

        # Предсказание
        y_pred = X @ self.coef_ + self.intercept_

        return y_pred

    def get_params(self, deep: bool = True) -> dict:
        """Получение параметров модели."""
        return {'fit_intercept': self.fit_intercept}

    def set_params(self, **params) -> 'CustomLinearReg':
        """Установка параметров модели."""
        for param, value in params.items():
            if param == 'fit_intercept':
                self.fit_intercept = value
            else:
                raise ValueError(f"Неизвестный параметр: {param}")
        return self

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Вычисление коэффициента детерминации R².
        
        Parameters
        ----------
        X : ArrayLike
            Матрица признаков
        y : ArrayLike
            Истинные значения
            
        Returns
        -------
        float
            R² score
        """
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot

    @property
    def features(self) -> Optional[np.ndarray]:
        """Получение имен признаков."""
        return self.feature_names_in_

    def __repr__(self) -> str:
        """Строковое представление модели."""
        return (f"CustomLinearReg(fit_intercept={self.fit_intercept})")

    def __str__(self) -> str:
        """Информативное строковое представление."""
        if self._is_fitted:
            return (
                f"CustomLinearReg(fit_intercept={self.fit_intercept}, "
                f"n_features={self.n_features_in_}, "
                f"fitted={self._is_fitted})"
            )
        return f"CustomLinearReg(fit_intercept={self.fit_intercept}, fitted=False)"


# Дополнительная утилита для кросс-валидации и оценки модели
class LinearRegressionCV:
    """
    Линейная регрессия с кросс-валидацией для диагностики.
    """
    @staticmethod
    def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисление среднеквадратичной ошибки."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисление средней абсолютной ошибки."""
        return np.mean(np.abs(y_true - y_pred))


# Создание синтетических данных
np.random.seed(42)
n_samples = 100
n_features = 3

X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 0.5])
true_intercept = 1.0
y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.1

# Разделение на train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели
model = CustomLinearReg(fit_intercept=True)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка качества
print(f"R² score: {model.score(X_test, y_test):.4f}")
print(f"Коэффициенты: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Число обусловленности: {model.condition_number_:.2e}")

# Сравнение с истинными значениями
print("\nСравнение с истинными параметрами:")
print(f"True coefficients: {true_coef}")
print(f"True intercept: {true_intercept}")