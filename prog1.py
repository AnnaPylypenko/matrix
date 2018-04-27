def mean(numbers):
    return float(sum(numbers))/len(numbers)
print('введите числа:')
nums=map(int, input().split())
l=list(nums)
print(l, type(l))
print("среднее =" +str(mean(l)))

l = [1, 2, 3]

for e in l:
  print(e)

  m = np.random.randint(1, 20, (10, 10))


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


  round(np.linalg.det(m)) == deternminant(m.tolist())

  m.tolist()