#%%
# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
# Загрузка датасета
data = pd.read_csv("winequality-red.csv")
#%%
# 1. Начальный анализ и очистка данных
# Просмотр первых нескольких строк
print(data.head())

# Проверка пропущенных значений
print(data.isnull().sum())

# Проверка типов данных
print(data.dtypes)

# Удаление дубликатов
data = data.drop_duplicates()

# Изучение статистических характеристик данных
print(data.describe())
#%%
# 2. Детальный анализ (EDA)
# Гистограмма целевой переменной
sns.histplot(data['quality'], bins=6, kde=False)
plt.show()

# Ящик с усами для признаков
sns.boxplot(data=data, orient='h')
plt.show()

# Корреляционная матрица
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
#%%
# 3. Визуализация важных переменных
# Пример: Диаграмма рассеяния для двух переменных
sns.scatterplot(x='alcohol', y='quality', data=data)
plt.show()
#%%
# 4. Выдвижение и проверка гипотез (пример)
# Гипотеза: Есть ли различие в качестве вина между красными и белыми винами?

# Пример: сравнение качества вина для разных уровней fixed acidity
red_wine_quality = data[data['quality'] >= 6]['fixed acidity']
white_wine_quality = data[data['quality'] < 6]['fixed acidity']

t_stat, p_value = stats.ttest_ind(red_wine_quality, white_wine_quality)
print(f"T-статистика: {t_stat}, p-значение: {p_value}")
#%%
# 5. Выбор типа регрессии и построение предсказаний (пример)
# Предположим, что вы хотите предсказать качество вина (quality)
X = data.drop(['quality'], axis=1)
y = data['quality']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Использование линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка производительности модели
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")
