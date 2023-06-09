# Efficient Homomorphic Comparison Methods with Optimal Complexity

a = 0.000001
b = 0

print(max(a, b))


def my_sqrt(x, d):
    a = x
    b = x - 1
    for n in range(d - 1):
        a = a * (1 - b / 2)
        b = b ** 2 * ((b - 3) / 4)
    return a


def my_max(a, b, d):
    x = (a + b) / 2
    y = (a - b) / 2
    z = my_sqrt(y ** 2, d)
    return x + z


print(my_max(a, b, d=100))
