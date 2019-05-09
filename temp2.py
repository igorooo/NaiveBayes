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
    X_train = X_train.T
    Dist = X @ X_train
    Dist2 = (~X) @ (~X_train)
    print('$$$$$')
    print(Dist)
    print('$$$$$')
    print(Dist2)
    return Dist



X = [[2,0,0,1],[0,1,1,0]]

X_train = [[0,0,0,1],[1,1,1,0],[1,1,1,1]]

X = np.array(X)
X_train = np.array(X_train)

print(X)
print('############')
print(~X)
print('############')
print(X_train)

print('############')


print(hamming_distance(X,X_train))
