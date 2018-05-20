
# coding: utf-8

# # 100 numpy exercises
# 
# This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow and in the numpy documentation. The goal of this collection is to offer a quick reference for both old and new users but also to provide a set of exercises for those who teach.
# 
# 
# If you find an error or think you've a better way to solve some of them, feel free to open an issue at <https://github.com/rougier/numpy-100>

# #### 1. Import the numpy package under the name `np` (★☆☆)

# In[4]:


import numpy as np


# #### 2. Print the numpy version and the configuration (★☆☆)

# In[3]:


print(np.__version__)
np.show_config()


# #### 3. Create a null vector of size 10 (★☆☆)

# In[1]:


Z = np.zeros(10)
print(Z)


# #### 4.  How to find the memory size of any array (★☆☆)

# In[4]:


Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))


# #### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)

# In[7]:


get_ipython().run_line_magic('run', '`python -c "import numpy; numpy.info(numpy.add)"`')


# #### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)

# In[5]:


Z = np.zeros(10)
Z[4] = 1
print(Z)


# #### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)

# In[6]:


Z = np.arange(10,50)
print(Z)


# #### 8.  Reverse a vector (first element becomes last) (★☆☆)

# In[44]:


Z = np.arange(50)
print(Z)
Z = Z[::-1]
print(Z)


# #### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)

# In[49]:


Z = np.arange(9).reshape(3,3)
print(Z)


# #### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)

# In[7]:


nz = np.nonzero([1,2,0,0,4,0])
print(nz)


# #### 11. Create a 3x3 identity matrix (★☆☆)

# In[8]:


Z = np.eye(3)
print(Z)


# #### 12. Create a 3x3x3 array with random values (★☆☆)

# In[9]:


Z = np.random.random((3,3,3))
print(Z)


# #### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)

# In[32]:


Z = np.random.randint(10, size=(10,10))
print(Z)
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)


# #### 14. Create a random vector of size 30 and find the mean value (★☆☆)

# In[12]:


Z = np.random.random(30)
m = Z.mean()
print(m)


# #### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)

# In[41]:


Z = np.ones((10,10))
print(Z)
Z[1:-1,1:-1] = 0
print(Z)


# #### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)

# In[5]:


Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)


# #### 17. What is the result of the following expression? (★☆☆)

# ```python
# 0 * np.nan
# np.nan == np.nan
# np.inf > np.nan
# np.nan - np.nan
# 0.3 == 3 * 0.1
# ```

# In[13]:


print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)


# #### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

# In[14]:


Z = np.diag(1+np.arange(4),k=-1)
print(Z)


# #### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)

# In[15]:


Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)


# #### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?

# In[16]:


print(np.unravel_index(100,(6,7,8)))


# #### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

# In[17]:


Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)


# #### 22. Normalize a 5x5 random matrix (★☆☆)

# In[18]:


Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z- Zmin)/(Zmax - Zmin)
print(Z)


# #### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

# In[20]:


color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])


# #### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

# In[21]:


Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)


# #### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)

# In[23]:


Z = np.arange(12)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)


# #### 26. What is the output of the following script? (★☆☆)

# ```python
# # Author: Jake VanderPlas
# 
# print(sum(range(5),-1))
# from numpy import *
# print(sum(range(5),-1))
# ```

# In[24]:


print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))


# #### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# ```python
# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z
# ```

# In[29]:



2 << Z >> 2
Z <- Z




# #### 28. What are the result of the following expressions?

# ```python
# np.array(0) / np.array(0)
# np.array(0) // np.array(0)
# np.array([np.nan]).astype(int).astype(float)
# ```

# In[30]:


np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)


# #### 29. How to round away from zero a float array ? (★☆☆)

# In[31]:


Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))


# #### 30. How to find common values between two arrays? (★☆☆)

# In[33]:


Z1 = np.random.randint(0,10,10)
print(Z1)
Z2 = np.random.randint(0,10,10)
print(Z2)
print(np.intersect1d(Z1,Z2))


# #### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

# In[35]:


defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0
_ = np.seterr(**defaults)


# #### 32. Is the following expressions true? (★☆☆)

# ```python
# np.sqrt(-1) == np.emath.sqrt(-1)
# ```

# In[39]:


np.sqrt(-1) == np.emath.sqrt(-1)
# print(np.sqrt(-1))
# print(np.emath.sqrt(-1))


# #### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

# In[43]:


yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday,today,tomorrow)


# #### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

# In[44]:


Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)


# #### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)

# In[48]:


A = np.ones(3)
B = np.ones(3)
C = np.ones(3)
print(A,B,C)
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)


# #### 36. Extract the integer part of a random array using 5 different methods (★★☆)

# In[6]:


Z = np.random.uniform(0,10,10)

print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))


# #### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

# In[7]:


Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)


# #### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

# In[8]:


def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)


# #### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

# In[9]:


Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)


# #### 40. Create a random vector of size 10 and sort it (★★☆)

# In[10]:


Z = np.random.random(10)
Z.sort()
print(Z)


# #### 41. How to sum a small array faster than np.sum? (★★☆)

# In[11]:


Z = np.arange(10)
np.add.reduce(Z)


# #### 42. Consider two random array A and B, check if they are equal (★★☆)

# In[12]:


A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)

# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)


# #### 43. Make an array immutable (read-only) (★★☆)

# In[13]:


Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1


# #### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

# In[14]:


Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)


# #### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

# In[15]:


Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)


# #### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)

# In[16]:


Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)


# ####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))

# In[17]:


X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))


# #### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)

# In[18]:


for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)


# #### 49. How to print all the values of an array? (★★☆)

# In[25]:


np.set_printoptions(threshold=np.nan)
Z = np.zeros((4,4))
print(Z)


# #### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

# In[27]:


Z = np.arange(10)
v = np.random.uniform(0,10)
index = (np.abs(Z-v)).argmin()
print(Z[index])


# #### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

# In[28]:


Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)


# #### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

# In[30]:


Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)


# #### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?

# #### 54. How to read the following file? (★★☆)

# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

# #### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

# #### 56. Generate a generic 2D Gaussian-like array (★★☆)

# In[31]:


X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)


# #### 57. How to randomly place p elements in a 2D array? (★★☆)

# In[32]:


n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)


# #### 58. Subtract the mean of each row of a matrix (★★☆)

# In[33]:


X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)
print(Y)


# #### 59. How to sort an array by the nth column? (★★☆)

# #### 60. How to tell if a given 2D array has null columns? (★★☆)

# #### 61. Find the nearest value from a given value in an array (★★☆)

# #### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

# #### 63. Create an array class that has a name attribute (★★☆)

# #### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

# #### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

# #### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)

# #### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

# #### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

# #### 69. How to get the diagonal of a dot product? (★★★)

# #### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

# #### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)

# #### 72. How to swap two rows of an array? (★★★)

# #### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

# In[34]:


faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)


# #### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

# In[39]:


C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print (A)


# #### 75. How to compute averages using a sliding window over an array? (★★★)

# In[40]:


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
Z = np.arange(20)
print(moving_average(Z, n=3))


# #### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)

# In[ ]:


from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)


# #### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

# In[43]:


Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z)


# #### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)

# #### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)

# #### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

# #### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)

# In[50]:


Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)


# #### 82. Compute a matrix rank (★★★)

# In[51]:


Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) 
rank = np.sum(S > 1e-10)
print(rank)


# #### 83. How to find the most frequent value in an array?

# In[52]:


Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())


# #### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

# In[53]:


Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)


# #### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)

# In[54]:


class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)


# #### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

# In[56]:


Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S)


# #### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

# #### 88. How to implement the Game of Life using numpy arrays? (★★★)

# #### 89. How to get the n largest values of an array (★★★)

# In[57]:


Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
print (Z[np.argpartition(-Z,n)[:n]])


# #### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)

# #### 91. How to create a record array from a regular array? (★★★)

# In[58]:


Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T, 
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)


# #### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

# #### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

# In[59]:


A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)


# #### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)

# #### 95. Convert a vector of ints into a matrix binary representation (★★★)

# #### 96. Given a two dimensional array, how to extract unique rows? (★★★)

# #### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

# In[61]:


A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B)
np.einsum('i,j->ij', A, B)    # np.outer(A, B)


# #### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

# #### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

# #### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
