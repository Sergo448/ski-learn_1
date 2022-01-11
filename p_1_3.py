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

z = knn.fit(X_train, y_train)
print(z)

KNC(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
    n_jobs=None, n_neighbors=1, p=2, weights='uniform')

X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма массива X_new: {}".format(X_new.shape))

pr = knn.predict(X_new)
pr_1 = knn.predict(X_test)
print("Прогноз вида на тестовом наборе:\n {}".format(pr_1))
print("Точность прогноща на тестовом наборе: {:.2f}".format(np.mean(pr_1 == y_test)))

print("Метка сорта цветка: {}".format(pr))
print("Сорт цветка: {}".format(iris_dataset['target_names'][pr]))

