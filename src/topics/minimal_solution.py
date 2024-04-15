import numpy as np
# create ndarray from these data:
# −1 1 3
# −2 0 2
# 3 1 −1

A = np.array([[-1, 1, 3], [-2, 0, 2], [3, 1, -1]])

def b(lam: float):
    return np.array([lam, 3 - 2 * lam, lam])

# mul A by b and print its norm for lam between -5 and 5
x = -1
while x <= 2:
    print(x, np.linalg.norm(b(x)))
    x += 0.1

