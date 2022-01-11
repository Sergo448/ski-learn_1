# Модуль SciLearn

""" Импортрруем все необходимые библиотеки"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier as KNC

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np

""" Импортировали все необходимые библиотеки"""

iris_dataset = load_iris()

# Получаем информацию о массиве данных
print('Ключи iris_dataset: \n{}'.format(iris_dataset.keys()))
print('Тип массива data: {}'.format(type(iris_dataset['data'])))
print('Форма массива data: {}'.format(iris_dataset['data'].shape))
print('Цель: {}'.format(iris_dataset['target']))
print('Названия ответов: {}'.format(iris_dataset['target_names']))
print(iris_dataset['DESCR'][:193] + "\n...")
print('Навзания признаков: \n{}'.format(iris_dataset['feature_names']))
print('Разположегие файла: \n{}'.format(iris_dataset['filename']))

# Пример
"""
# Пример признаков первых 5 примеров
print("Первые 5 строк массива data:\n{}".format(iris_dataset['data'][:5]))

# Получение значений выбранного массива
print('Правильные ответы:\n{}'.format(iris_dataset['target']))
"""

# Формирование обучающей и тестирующей выборки
X_train, X_test, y_train, y_test = tts(iris_dataset['data'],
                                       iris_dataset['target'],
                                       random_state=0)

print('Размерность массива X_train: {}'.format(X_train.shape))
print('Размерность массива y_train: {}'.format(y_train.shape))
print('Размерность массива X_test: {}'.format(X_test.shape))
print('Размерность массива y_test: {}'.format(y_test.shape))

# Создание и обучение классификатора

knn = KNC(n_neighbors=1)
knn.fit(X_train, y_train)

# Практическое использованме классификатора
X_new = np.array([[5, 2.9, 1, 0.2]])
pr = knn.predict(X_new)
# print("Метка вида цветка: {}".formst(pr))
print("Вид цветка: {}".format(iris_dataset['target_names'][pr]))

pr = knn.predict(X_test)
print("Прогноз вида на тест овом наборе:\n {}".format(pr))
print("Точность проноза на тестовом наборе: {:.2f}".format(np.mean(pr == y_test)))

# Матрица рассеяния с библиотекой seaborn
df = sb.load_dataset('iris')
sb.set_style("ticks")
sb.pairplot(df, hue='species', diag_kind='kde', kind='scatter', palette='husl')
plt.show()


