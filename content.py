# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np

# x,y - vectors


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
    X = X.toarray()
    X_train = X_train.toarray()
    N1, D = np.shape(X)
    N2, _ = np.shape(X_train)

    X_train = X_train.T
    Dist = X.astype(int) @ X_train.astype(int)
    Dist += (~X).astype(int) @ (~X_train).astype(int)
    Dist = np.ones((N1, N2)) * D - Dist
    return Dist


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    labels = Dist.argsort(kind='mergesort')
    return y[labels]


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


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    N, num_of_classes = np.shape(p_y_x)

    P_y_x = np.fliplr(p_y_x)
    predict = num_of_classes - np.argmax(P_y_x, axis=1) - 1  # couse of flipped rows
    dif = predict - y_true
    return np.count_nonzero(dif) / np.shape(y_true)[0]


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """

    h_dist = hamming_distance(X_val, X_train)
    labels = sort_train_labels_knn(h_dist, y_train)

    errors = []
    curr_best = 0

    for i in range(0, len(k_values)):
        p_y_x = p_y_x_knn(labels, k_values[i])
        err = classification_error(p_y_x, y_val)
        errors.append(err)

        if err < errors[curr_best]:
            curr_best = i
    #best = np.argmin(errors)
    return errors[curr_best], k_values[curr_best], errors


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    print('6')
    return np.bincount(y_train) / len(y_train)


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    print('7')
    num_of_classes = 4

    # how many appears of word d in every appear of class K
    N, D = np.shape(X_train)
    result = np.zeros((num_of_classes, D))

    for i in range(0, N):
        result[y_train[i]] += X_train[i]

        """for j in range(0, D):
            if X_train[i, j] == 1:
                result[y_train[i], j] += 1
        """

    result = result + a - 1

    devider = np.bincount(y_train, None, num_of_classes) + a + b - 2
    """
    print('@@@@@@@@@@@############')
    print(result)
    print('^^^^^^^^^^^^^^^^')
    print(devider)
    """

    # for i in range(0, num_of_classes):
    #result[i] = result[i] / devider[i]
    result = np.divide(result, devider.reshape(4, 1))

    return result


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    print('8')

    num_of_classes = 4
    N, D = np.shape(X)
    X = X.toarray()
    p_x_1_y_rev = 1 - p_x_1_y
    X_rev = 1 - X
    result = []

    for i in range(0, N):
        # p(x | y)  Bernoulli wyjaśnione w zeszycie
        p = np.prod((p_x_1_y ** X[i, ]) * (p_x_1_y_rev ** X_rev[i, ]), axis=1) * p_y
        sum = np.sum(p)  # prawd całkowite
        result.append(p / sum)
    return result


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.

    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """

    N2, D = np.shape(X_train)
    N1, _ = np.shape(X_val)

    # ilosc parametrow
    a_len = np.shape(a_values)[0]
    b_len = np.shape(b_values)[0]

    a_priori = estimate_a_priori_nb(y_train)
    errors = np.zeros((a_len, b_len))
    best_err = np.inf
    best_a = 0
    best_b = 0

    for i in range(0, a_len):
        for j in range(0, b_len):
            p_x_y = estimate_p_x_y_nb(X_train, y_train, a_values[i], b_values[j])
            p_y_x = p_y_x_nb(a_priori, p_x_y, X_val)
            err = classification_error(p_y_x, y_val)
            errors[i, j] = err

            if err < best_err:
                best_a = i
                best_b = j
                best_err = err

    return best_err, a_values[best_a], b_values[best_b], errors
