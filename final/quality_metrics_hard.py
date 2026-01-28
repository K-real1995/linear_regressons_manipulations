# Импорт всех необходимых библиотек
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================

print("=" * 60)
print("ШАГ 1: Загрузка California Housing Dataset")
print("=" * 60)

# Загружаем датасет о ценах на жилье в Калифорнии
housing = fetch_california_housing()
X = housing.data  # Матрица признаков (фичей)
y = housing.target  # Вектор целевой переменной (цены)

print(f"Названия признаков: {housing.feature_names}")
print(f"Форма данных X: {X.shape}")
print(f"Форма целевой переменной y: {y.shape}")
print(f"\nОписание признаков:")
for i, feature in enumerate(housing.feature_names):
    print(f"{i+1}. {feature}")

# ============================================
# 2. РАЗДЕЛЕНИЕ ДАННЫХ НА TRAIN И VALID
# ============================================

print("\n" + "=" * 60)
print("ШАГ 2: Разделение данных на обучающую и валидационную выборки")
print("=" * 60)

# Разделяем данные в соотношении 80% train / 20% valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,      # 20% данных в валидационную выборку
    random_state=42,    # Фиксируем seed для воспроизводимости
    shuffle=True        # Перемешиваем данные перед разделением
)

print("Размеры выборок после разделения:")
print(f"Обучающая выборка (train): X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Валидационная выборка (valid): X_valid = {X_valid.shape}, y_valid = {y_valid.shape}")
print(f"\nПроцентное соотношение:")
print(f"Train: {len(X_train) / len(X) * 100:.1f}% данных")
print(f"Valid: {len(X_valid) / len(X) * 100:.1f}% данных")

# ============================================
# 3. ОБУЧЕНИЕ МОДЕЛИ НА НОРМАЛЬНЫХ ДАННЫХ
# ============================================

print("\n" + "=" * 60)
print("ШАГ 3: Обучение модели линейной регрессии на исходных данных")
print("=" * 60)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказания на валидационной выборке
y_pred = model.predict(X_valid)

# Вычисляем коэффициент детерминации R²
r2_original = r2_score(y_valid, y_pred)

print("Результаты обучения на исходных данных:")
print(f"Коэффициент детерминации R² = {r2_original:.6f}")
print(f"Свободный член (intercept) = {model.intercept_:.6f}")
print(f"\nКоэффициенты модели (веса признаков):")
for i, (feature, coef) in enumerate(zip(housing.feature_names, model.coef_)):
    print(f"{feature:>15}: {coef:>10.6f}")

# ============================================
# 4. ПРИМЕНЕНИЕ Z-ПРЕОБРАЗОВАНИЯ (СТАНДАРТИЗАЦИИ)
# ============================================

print("\n" + "=" * 60)
print("ШАГ 4: Применение Z-преобразования (стандартизации)")
print("=" * 60)

# Z-преобразование: (x - mean) / std
# Делаем признаки с нулевым средним и единичной дисперсией

scaler = StandardScaler()

# Важно: обучаем scaler ТОЛЬКО на обучающих данных
# чтобы избежать "утечки" информации из валидационной выборки
X_train_scaled = scaler.fit_transform(X_train)

# Преобразуем валидационные данные тем же scaler'ом
X_valid_scaled = scaler.transform(X_valid)

print("Статистики до стандартизации (первые 5 признаков):")
print(f"{'Признак':>15} | {'Среднее (train)':>15} | {'Стд. отклонение (train)':>20}")
print("-" * 60)
for i in range(5):
    print(f"{housing.feature_names[i]:>15} | {X_train[:, i].mean():>15.3f} | {X_train[:, i].std():>20.3f}")

print(f"\nСтатистики после стандартизации (первые 5 признаков):")
print(f"{'Признак':>15} | {'Среднее (train)':>15} | {'Стд. отклонение (train)':>20}")
print("-" * 60)
for i in range(5):
    print(f"{housing.feature_names[i]:>15} | {X_train_scaled[:, i].mean():>15.6f} | {X_train_scaled[:, i].std():>20.6f}")

print(f"\nПроверка для валидационной выборки (первые 5 признаков):")
print(f"{'Признак':>15} | {'Среднее (valid)':>15} | {'Стд. отклонение (valid)':>20}")
print("-" * 60)
for i in range(5):
    print(f"{housing.feature_names[i]:>15} | {X_valid_scaled[:, i].mean():>15.6f} | {X_valid_scaled[:, i].std():>20.6f}")

# ============================================
# 5. ОБУЧЕНИЕ МОДЕЛИ НА СТАНДАРТИЗИРОВАННЫХ ДАННЫХ
# ============================================

print("\n" + "=" * 60)
print("ШАГ 5: Обучение модели на стандартизированных данных")
print("=" * 60)

# Создаем новую модель для стандартизированных данных
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

# Делаем предсказания на стандартизированной валидационной выборке
y_pred_scaled = model_scaled.predict(X_valid_scaled)

# Вычисляем R² для стандартизированных данных
r2_scaled = r2_score(y_valid, y_pred_scaled)

print("Результаты обучения на стандартизированных данных:")
print(f"Коэффициент детерминации R² = {r2_scaled:.6f}")
print(f"Свободный член (intercept) = {model_scaled.intercept_:.6f}")
print(f"\nКоэффициенты модели (веса признаков):")
for i, (feature, coef) in enumerate(zip(housing.feature_names, model_scaled.coef_)):
    print(f"{feature:>15}: {coef:>10.6f}")

# ============================================
# 6. СРАВНЕНИЕ РЕЗУЛЬТАТОВ И ВЫВОДЫ
# ============================================

print("\n" + "=" * 60)
print("ШАГ 6: Сравнение результатов и анализ")
print("=" * 60)

print("\nСРАВНЕНИЕ МЕТРИК R²:")
print("-" * 40)
print(f"Без стандартизации: {r2_original:.6f}")
print(f"Со стандартизацией:  {r2_scaled:.6f}")
print(f"Разница: {r2_scaled - r2_original:+.6f}")

print("\nСРАВНЕНИЕ КОЭФФИЦИЕНТОВ МОДЕЛИ:")
print("-" * 40)
print(f"{'Признак':>15} | {'Без стан-ии':>12} | {'Со стан-ей':>12} | {'Отношение':>12}")
print("-" * 60)
for i, feature in enumerate(housing.feature_names):
    ratio = model_scaled.coef_[i] / model.coef_[i] if model.coef_[i] != 0 else np.inf
    print(f"{feature:>15} | {model.coef_[i]:>12.6f} | {model_scaled.coef_[i]:>12.6f} | {ratio:>12.6f}")

print("\nСРАВНЕНИЕ СВОБОДНЫХ ЧЛЕНОВ:")
print("-" * 40)
print(f"Без стандартизации: intercept = {model.intercept_:.6f}")
print(f"Со стандартизацией: intercept = {model_scaled.intercept_:.6f}")

print("\n" + "=" * 60)
print("ТЕОРЕТИЧЕСКОЕ ОБЪЯСНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

print("""
ДЛЯ ЛИНЕЙНОЙ РЕГРЕССИИ:

1. Математическая модель: y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

2. При стандартизации признаков:
   xᵢ' = (xᵢ - μᵢ) / σᵢ
   
   Подставляем в модель:
   y = w₁(σ₁x₁' + μ₁) + w₂(σ₂x₂' + μ₂) + ... + wₙ(σₙxₙ' + μₙ) + b
     = (w₁σ₁)x₁' + (w₂σ₂)x₂' + ... + (wₙσₙ)xₙ' + (w₁μ₁ + w₂μ₂ + ... + wₙμₙ + b)

3. Новые коэффициенты:
   wᵢ' = wᵢσᵢ
   b' = w₁μ₁ + w₂μ₂ + ... + wₙμₙ + b

4. ПРЕДСКАЗАНИЯ ОСТАЮТСЯ ТОЧНО ТАКИМИ ЖЕ:
   y_pred = w₁x₁ + ... + b = w₁'x₁' + ... + b'
   
5. Следовательно, R² score НЕ ДОЛЖЕН ИЗМЕНЯТЬСЯ.

Практически видимые небольшие отличия (обычно < 10⁻¹⁴) вызваны:
- Численной погрешностью вычислений с плавающей точкой
- Особенностями реализации алгоритма в sklearn
- Округлением на разных этапах обработки
""")

print("\n" + "=" * 60)
print("ВЫВОДЫ")
print("=" * 60)

if abs(r2_scaled - r2_original) < 1e-10:
    print("✓ Метрика R² практически не изменилась (разница < 10⁻¹⁰)")
    print("✓ Это подтверждает теоретическое положение об инвариантности")
    print("  линейной регрессии к масштабированию признаков")
else:
    print(f"⚠ Обнаружена заметная разница: {r2_scaled - r2_original:.2e}")
    print("  Это может быть вызвано:")
    print("  - Ошибками в реализации")
    print("  - Особенностями конкретного датасета")
    print("  - Численной нестабильностью")

print("\n" + "=" * 60)
print("ЗАЧЕМ НУЖНА СТАНДАРТИЗАЦИЯ В ЛИНЕЙНОЙ РЕГРЕССИИ?")
print("=" * 60)

print("""
Хотя R² не меняется, стандартизация полезна для:

1. ИНТЕРПРЕТАЦИИ КОЭФФИЦИЕНТОВ:
   - После стандартизации коэффициенты показывают, на сколько изменится y
     при изменении признака на 1 стандартное отклонение
   - Коэффициенты становятся сравнимыми между собой

2. РЕГУЛЯРИЗАЦИИ:
   - Методы Ridge и Lasso чувствительны к масштабу признаков
   - Без стандартизации регуляризация будет неравномерной

3. СКОРОСТИ СХОДИМОСТИ:
   - Для градиентного спуска стандартизация ускоряет обучение

4. УСТОЙЧИВОСТИ К ВЫБРОСАМ:
   - Стандартизация делает алгоритм более устойчивым
""")

# Дополнительная проверка: сравним предсказания напрямую
print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА")
print("=" * 60)

# Вычислим максимальную абсолютную разницу между предсказаниями
max_diff = np.max(np.abs(y_pred - y_pred_scaled))
mean_diff = np.mean(np.abs(y_pred - y_pred_scaled))

print(f"Максимальная разница между предсказаниями: {max_diff:.2e}")
print(f"Средняя разница между предсказаниями: {mean_diff:.2e}")

if max_diff < 1e-10:
    print("✓ Предсказания практически идентичны")
else:
    print("⚠ Обнаружены различия в предсказаниях")