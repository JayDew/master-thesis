import numpy as np


def import_data(index):
    def fun2():
        c = np.asarray([2, 2])
        A = np.asarray([[1, 1]])
        b = np.asarray([3])
        return c, A, b

    def fun3():
        c = np.asarray([-2, 2, -1])
        A = np.asarray([[1, 1, 1]])
        b = np.asarray([2])
        return c, A, b

    def fun4():
        # Taken from https://www.cuemath.com/algebra/linear-programming/
        c = np.asarray([5, 4, 0, 0])
        A = np.asarray([[4, 1, -1, 0], [2, 3, 0, -1]])
        b = np.asarray([40, 90])
        return c, A, b

    def fun5():
        # taken from AA slides, lecture 7
        c = np.asarray([2, 4, 1, 5, 3])
        A = np.asarray([[1, 1, 0, 0, 0], [-1, 0, 1, 1, 0], [0, -1, -1, 0, 1], [0, 0, 0, -1, -1]])
        b = np.asarray([1, 0, 0, -1])
        return c, A, b

    def fun6():
        # taken from AA slides
        c = np.asarray([3, 3, 4, 0, 0, 0])
        A = np.asarray([[2, 1, 0, -1, 0, 0], [0, 1, 4, 0, -1, 0], [4, 0, 8, 0, 0, -1]])
        b = np.asarray([3, 2, 9])
        return c, A, b

    data = [None, None, fun2, fun3, fun4, fun5, fun6]
    return data[index]()
