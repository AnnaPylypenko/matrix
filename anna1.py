import numpy as np
import matplotlib.pyplot as plt
import copy


# Инициализация исходных матриц
m = np.random.randint(1, 20, (3, 3)).tolist()
m1 = copy.deepcopy(m)
m2 = copy.deepcopy(m)
print('Исходная матрица')
print(m)


def minor(m, i, j):
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def deternminant(m):
    # base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * deternminant(minor(m, 0, c))
    return determinant


a = deternminant(m)
print('детерминант исходной матрицы:')
print(a)

# Создаем матрицу миноров m1
for i in range(len(m)):
    for j in range(len(m)):
        m1[i][j] = ((-1)**(i+j)) * deternminant(minor(m, i, j))

print(m1)

# Вычисляем обратную матрицу m2
if a != 0:
    for i in range(len(m)):
        for j in range(len(m)):
            m2[i][j] = m1[j][i] / a
    print('Обратная матрица')
    for row in m2:
        for e in row:
            print("%.2f" % e, end='\t')
        print()
    # print(m2)
else:
   print('Обратной матрицы не существует')
