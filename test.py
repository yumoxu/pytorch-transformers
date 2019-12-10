import numpy as np

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

a = a.reshape(3, 1)
b = b.reshape(1, 3)

print(np.dot(a, b))
