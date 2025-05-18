import numpy as np
a = np.zeros(3, dtype=[('a', 'f4'), ('b', 'i4'), ('c', 'S5')])
b = np.zeros(3, dtype=[('a', 'f2'), ('b', 'i2'), ('c', 'S2')])

# a, b 배열 재정의
a = np.zeros(13, dtype=[('a', 'f4'), ('b', 'i4'), ('c', 'S7')])
b = np.zeros(10, dtype=[('a', 'f2'), ('b', 'i2'), ('c', 'S2')])
for i in range(13): a[i] = (0.5 * (i + 1), 2 * (i + 1), b'a' * (i + 1 if i <= 4 else 5 if i <= 6 else 6 if i == 7 else 7))
for i in range(10): b[i] = ((i + 1) ** 2, (i+1)**5, bytes(chr(97 + i), 'utf-8'))

print(a)
print(b)