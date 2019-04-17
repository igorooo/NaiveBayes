import numpy as np


def merge(a, b):
    """ a = b = touple (value,label) """
    c = []
    while len(a) != 0 and len(b) != 0:
        if a[0][0] < b[0][0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])
    if len(a) == 0:
        c += b
    else:
        c += a
    return c

# X-vector of touples(value,label)


def mergesort(x):
    """ returns matrix of touples (value,label) """
    if len(x) == 0 or len(x) == 1:
        return x
    else:
        middle = int(len(x)/2)
        a = mergesort(x[:middle])
        b = mergesort(x[middle:])

    return merge(a, b)


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.

            print("////////------------///////////\n\n\n\n\n\n\n\n")
            print(y_sorted)
            print("////////------------///////////\n\n\n\n\n\n\n\n")
            print(y_sorted_expected)
    """

    N1, N2 = np.shape(Dist)
    res = np.zeros((N1, N2))

    for n1 in range(0, N1):
        temp = mergesort(list(zip(Dist[n1], y)))
        for n2 in range(0, N2):
            res[n1, n2] = temp[n2][1]

    return res


D = np.array([[2, 5, 3], [6, 7, 1]])
y = np.array([0, 3, 2])

print(D ** y)
