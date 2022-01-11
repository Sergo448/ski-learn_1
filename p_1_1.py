import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import pandas as pd

# Создаем 2Д массив NumPy с единицами по главной диагонали
# и нулями в остальных ячейках
eye = np.eye(4)
print("Массив NumPy:\n{}".format(eye))

# Преобразовываем массив NumPy в разряженную мат рицу SciPy в формате CSR
sparse_matrix = sparse.csr_matrix(eye)
print("\nразреженная матрица SciPy в формате CSR:\n{}".format(sparse_matrix))

# Разреженая матрица формата COO
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('формат COO:\n{}'.format(eye_coo))

# График зависимости 2-х величин

x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker='.')
plt.show()

data = {
    'Имя': ["Дима", "Анна", " Петр", "Вика"],
    "Город": ["Москва", "Курск", "Псков", "Воронеж"],
    "Возраст": [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)
print(data_pandas)