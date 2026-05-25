import numpy as np
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

# Генерируем случайные матрицы
A = np.random.rand(n, n)
B = np.random.rand(n, n)

with open('matrixA.txt', 'w') as f:
    f.write(f"{n}\n")
    for row in A:
        f.write(" ".join(map(str, row)) + "\n")

with open('matrixB.txt', 'w') as f:
    f.write(f"{n}\n")
    for row in B:
        f.write(" ".join(map(str, row)) + "\n")

print(f"Созданы матрицы {n}x{n}")