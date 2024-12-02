import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

# URL для загрузки данных (данные о ценах на жильё в Бостоне)
data_url = "http://lib.stat.cmu.edu/datasets/boston"

# Чтение данных, начиная с 23-й строки (skiprows=22), с использованием регулярного выражения для разделителя
raw_data = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

# Разделение данных на две части:
# - Чётные строки (индексы 0, 2, 4, ...) содержат основные признаки
even_rows = raw_data.iloc[::2, :]

# - Нечётные строки (индексы 1, 3, 5, ...) содержат оставшиеся признаки и целевую переменную
odd_rows = raw_data.iloc[1::2, :]

# Соединение признаков: 
# - Признаки из чётных строк (все столбцы)
# - Первые два столбца из нечётных строк
features = np.hstack([even_rows.values, odd_rows.iloc[:, :2].values])

# Выделение целевой переменной (третий столбец из нечётных строк)
target = odd_rows.iloc[:, 2].values

# Вывод формы массивов для проверки
print(f"Форма признаков (features): {features.shape}")  # Ожидается (506, 13)
print(f"Форма целевой переменной (target): {target.shape}")  # Ожидается (506,)

# Преобразование в DataFrame для удобства работы
X = pd.DataFrame(features)
Y = pd.DataFrame(target)

# Описание статистик признаков и целевой переменной
print(X.describe())
print(Y.describe())

# Построение столбчатой диаграммы для признаков
X.plot.bar(stacked=True)

# Построение тепловой карты для корреляционной матрицы признаков
plt.figure(figsize=(10,7))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Масштабирование данных
scaler = StandardScaler()

# Обучаем StandardScaler на обучающих данных и масштабируем обучающую выборку
X_train = scaler.fit_transform(X_train)

# Применяем масштабирование к тестовой выборке
X_test = scaler.transform(X_test)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание значений на тестовой выборке
y_pred = model.predict(X_test)

# Построение столбчатой диаграммы для коэффициентов линейной регрессии
plt.figure(figsize=(20, 8))

# Если X.columns не имеет значений (например, индексные колонки или без имени), создайте список вручную
# Например:
features_names = [f'Feature {i+1}' for i in range(X.shape[1])]  # если у вас нет имен в X.columns

# Строим столбчатую диаграмму с коэффициентами
plt.bar(features_names, model.coef_.flatten())  # Используем .flatten() для преобразования в одномерный массив
plt.xticks(rotation=90)  # Поворачиваем метки на оси X для лучшего отображения
plt.title('Коэффициенты линейной регрессии')
plt.xlabel('Признаки')
plt.ylabel('Коэффициенты')
plt.show()

# Оценка модели на основе MSE и MAE
y_train_pred = model.predict(X_train)

# Вычисляем метрики для обучающей выборки
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Вычисляем метрики для тестовой выборки
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)

# Выводим результаты
print(f'Train MSE: {train_mse:.2f}, Train MAE: {train_mae:.2f}')
print(f'Test MSE: {test_mse:.2f}, Test MAE: {test_mae:.2f}')

# Среднее значение целевой переменной
print(f"Среднее значение целевой переменной: {Y.mean()}")

# Кросс-валидация модели
result = cross_val_score(estimator=LinearRegression(), X=X, y=Y, scoring='neg_mean_absolute_error', cv=5)
print(f'Среднее MAE равно {-result.mean():.2f}, стандартное отклонение MAE равно {result.std():.2f}')
