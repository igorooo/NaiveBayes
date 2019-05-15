import numpy as np


def h_distance(x, y):
    temp = 0
    for e1, e2 in zip(x, y):
        if e1 != e2:
            temp += 1
    return temp


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    #X = X.toarray()
    #X_train = X_train.toarray()
    N1, D = np.shape(X)
    N2, _ = np.shape(X_train)

    X_train = X_train.T
    Dist = X.astype(int) @ X_train.astype(int)
    Dist += (~X).astype(int) @ (~X_train).astype(int)
    Dist = np.ones((N1, N2)) * D - Dist
    return Dist


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """

    num_of_classes = 4
    result = []

    N1, _ = np.shape(y)
    for i in range(0, N1):
        temp = y[i, :k]
        part_result = np.bincount(temp, None, num_of_classes)
        result.append([part_result[j] / k for j in range(0, num_of_classes)])
    return result


X = [[1, 0, 0, 1, ], [0, 1, 1, 0]]

X_train = [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]

x = np.array([True, False, True, True])

X = np.array(X).astype(np.bool)
X_train = np.array(X_train).astype(np.bool)


print('############')
dist = hamming_distance(X, X_train)
print(dist)

y = np.array([0, 3, 2])

labels = dist.argsort(kind='mergesort')

y = y[labels]


print(y)
print('$$$$')
print(p_y_x_knn(y, 2))
print('$$$$')
