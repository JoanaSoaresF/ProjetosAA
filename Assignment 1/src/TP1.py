import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from sklearn import svm

test_file = r"C:\Users\joana\Documents\Escola\Engenharia Informática\4ºAno\1ºSemestre\Aprendizagem Automática\ProjetosAA\Assignment 1\data\TP1_test.tsv"
train_file = r"C:\Users\joana\Documents\Escola\Engenharia Informática\4ºAno\1ºSemestre\Aprendizagem Automática\ProjetosAA\Assignment 1\data\TP1_train.tsv"


def standardization(train, test):
    means = np.mean(train, axis=0)
    stds = np.std(train, axis=0)
    s_train = (train - means) / stds
    s_test = (test - means) / stds
    return s_train, s_test


def load_data(train_file, test_file):
    train = np.loadtxt(train_file, delimiter='\t')
    test = np.loadtxt(test_file, delimiter='\t')
    train = shuffle(train)
    test = shuffle(test)
    train_X = train[:, :-1]  # features of the training set
    train_Y = train[:, -1]  # classes of the training set
    test_X = test[:, :-1]  # features of the testing set
    test_Y = test[:, -1]  # classes of the testing set
    train_X, test_X = standardization(train_X, test_X)

    return train_X, train_Y, test_X, test_Y


def apriori_probability(data):
    """Returns the logarithm probability of the class 0 and class 1 from the data"""

    total = data.shape[0]  # número total de exemplos
    class0 = data[data[:] == 0].shape[0]  # número de exemplos da classe 0
    class1 = data[data[:] == 1].shape[0]  # número de exemplos da classe 1
    return np.log(class0 / total), np.log(class1 / total)


# bandwidth
# full training set ( X, Y) - training and validation
# indices para a parte de treino
# indices para a parte de validação
def naive_bayes_fold_errors(bandwidth, X, Y, train_ix, valid_ix):
    """return error for train and validation sets"""

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)

    train_set_X = X[train_ix, :]
    train_set_Y = Y[train_ix]
    apriori_class0, apriori_class1 = apriori_probability(train_set_Y)  # calculated only in the training set
    class_apriori_prob = [apriori_class0, apriori_class1]

    # probabilities calculated for all the X set, for each class
    classes_probability = np.zeros((X.shape[0], 2))
    classes = [0, 1]
    for c in classes:
        train_class_examples = train_set_X[train_set_Y == c]
        sum_probs = np.zeros(X.shape[0])
        for attribute in range(X.shape[1]):
            attribute_example = train_class_examples[:, [attribute]]  # [] faz coluna
            kde.fit(attribute_example)  # train only with the training examples
            sum_probs += kde.score_samples(X[:, [attribute]])  # score sample for the complete set
        classes_probability[:, c] = class_apriori_prob[c] + sum_probs  # store the probabilities calculated

    predictions = np.argmax(classes_probability, axis=1)  # chooses the class with maximum probability

    num_error_train = np.count_nonzero(np.not_equal(predictions[train_ix], Y[train_ix]))
    num_error_valid = np.count_nonzero(np.not_equal(predictions[valid_ix], Y[valid_ix]))
    total_train = train_ix.shape[0]
    total_valid = valid_ix.shape[0]
    return num_error_train / total_train, num_error_valid / total_valid


# full training set ( X_r, Y_r) - training and validation
# test set (X_t e Y_t)
def naive_bayes_test_error(bandwidth, train_X, train_Y, test_X, test_Y):
    """return test error and predictions on test set"""

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    # calculated only in the training set
    apriori_class0, apriori_class1 = apriori_probability(train_Y)

    class_apriori_prob = [apriori_class0, apriori_class1]

    # probabilities calculated for all the X set, for each class
    classes_probability = np.zeros((test_X.shape[0], 2))
    classes = [0, 1]
    for c in classes:
        train_class_examples = train_X[train_Y == c]
        sum_probs = np.zeros(test_X.shape[0])
        for attribute in range(test_X.shape[1]):
            attribute_example = train_class_examples[:, [attribute]]  # [] faz coluna
            kde.fit(attribute_example)  # train only with the training examples
            sum_probs += kde.score_samples(test_X[:, [attribute]])  # score sample for the complete set
        classes_probability[:, c] = class_apriori_prob[c] + sum_probs  # store the probabilities calculated

    predictions = np.argmax(classes_probability, axis=1)  # chooses the class with maximum probability
    num_error_test = np.count_nonzero(np.not_equal(predictions, test_Y))

    return num_error_test / test_Y.shape[0], predictions


def naive_bayes(train_X, train_Y, test_X, test_Y, folds):
    """Returns the best bandwidth, the predictions on the test set and the test error. Computed by the Naive Bayes
    classifier """
    kf = StratifiedKFold(n_splits=folds)
    lowest_error = 100000
    best_bandwidth = -1
    errs = []  # used to plot the error
    for bandwidth in np.arange(0.02, 0.62, 0.02):
        # measure which one is the best number of features
        train_error = val_error = 0
        # Split divide o set para treino (X_r e Y_r) em dois sets, uma para validação e outro para treino e tenta
        # manter a mesma proporção das classes e corremos para cada partição. Obtemos osn indices que vamos usar para
        # validação e treino
        for tr_ix, val_ix in kf.split(train_Y, train_Y):
            t, v = naive_bayes_fold_errors(bandwidth, train_X, train_Y, tr_ix, val_ix)
            train_error += t
            val_error += v

        # Média do erro de validação em cada partição. Dá-nos um melhor erro de validação, devido à média
        val_error = val_error / folds
        train_error = train_error / folds
        # print("Bandwidth:{}, training_error:{}, validation_error:{}".format(bandwidth, train_error, val_error))
        errs.append((train_error, val_error))
        if val_error < lowest_error:
            lowest_error = val_error
            best_bandwidth = bandwidth

    errs = np.array(errs)
    # Já sabemos qual o melhor modelo
    test_error, predictions = naive_bayes_test_error(best_bandwidth, train_X, train_Y, test_X, test_Y)
    # Plot
    plot(errs, "Train error", "Validation error", "Bandwidth (x100)", "Error", "Naive Bayes", range(2, 62, 2))
    return best_bandwidth, predictions, test_error


def plot(errs, train_label, validation_label, x_label, y_label, graphic_title, x_values):
    fig = plt.figure(figsize=(8, 8), frameon=True)
    plt.plot(x_values, errs[:, 0], '-b', linewidth=3, label=train_label)
    plt.plot(x_values, errs[:, 1], '-r', linewidth=3, label=validation_label)
    plt.title(graphic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper right")
    file_name = ""
    if "Naive" in graphic_title:
        file_name = "NB"
    else:
        file_name = "SVM"
    plt.savefig("../{}.png".format(file_name), dpi=300)
    plt.show()
    plt.close()


def svm_fold_errors(gamma, C, X, Y, tr_ix, val_ix):
    sv = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    sv.fit(X[tr_ix], Y[tr_ix])
    train_error = 1 - sv.score(X[tr_ix], Y[tr_ix])
    validation_error = 1 - sv.score(X[val_ix], Y[val_ix])  # returns mean accuracy, we want the the errors
    return train_error, validation_error


def svm_test_error(best_gamma, C, train_X, train_Y, test_X, test_Y):
    sv = svm.SVC(C=C, kernel='rbf', gamma=best_gamma)
    sv.fit(train_X, train_Y)
    predictions = sv.predict(test_X)
    num_error_test = np.count_nonzero(np.not_equal(predictions, test_Y))
    return num_error_test / test_Y.shape[0], predictions


def svm_classifier(train_X, train_Y, test_X, test_Y, folds):
    """Returns the best bandwidth, the predictions on the test set and the test error. Computed by the Naive Bayes
        classifier """
    kf = StratifiedKFold(n_splits=folds)
    lowest_error = 100000
    best_gamma = -1
    errs = []  # used to plot the error
    for gamma in np.arange(0.2, 6.2, 0.2):
        # measure which one is the best number of features
        train_error = val_error = 0
        # Split divide o set para treino (X_r e Y_r) em dois sets, uma para validação e outro para treino e tenta
        # manter a mesma proporção das classes e corremos para cada partição. Obtemos osn indices que vamos usar para
        # validação e treino
        for tr_ix, val_ix in kf.split(train_Y, train_Y):
            t, v = svm_fold_errors(gamma, 1, train_X, train_Y, tr_ix, val_ix)
            train_error += t
            val_error += v

        # Média do erro de validação em cada partição. Dá-nos um melhor erro de validação, devido à média
        val_error = val_error / folds
        train_error = train_error / folds
        # print("Gamma:{}, training_error:{}, validation_error:{}".format(gamma, train_error, val_error))
        errs.append((train_error, val_error))
        if val_error < lowest_error:
            lowest_error = val_error
            best_gamma = gamma

    errs = np.array(errs)
    # Já sabemos qual o melhor modelo
    test_error, predictions = svm_test_error(best_gamma, 1, train_X, train_Y, test_X, test_Y)
    # Plot
    plot(errs, "Train error", "Validation error", "Gamma (x10)", "Error", "SVM Classifier", range(2, 62, 2))
    return best_gamma, predictions, test_error


def svm_classifier_with_c(train_X, train_Y, test_X, test_Y, folds):
    """Returns the best bandwidth, the predictions on the test set and the test error. Computed by the Naive Bayes
        classifier """
    kf = StratifiedKFold(n_splits=folds)
    lowest_error = 100000
    best_gamma = -1
    best_c = -1
    errs = []  # used to plot the error
    for gamma in np.arange(0.2, 6.2, 0.2):
        # measure which one is the best number of features
        train_error = val_error = 0
        for exp in range(-2, 4):
            c = 10 ** exp
            for tr_ix, val_ix in kf.split(train_Y, train_Y):
                t, v = svm_fold_errors(gamma, c, train_X, train_Y, tr_ix, val_ix)
                train_error += t
                val_error += v

            # Média do erro de validação em cada partição. Dá-nos um melhor erro de validação, devido à média
            val_error = val_error / folds
            train_error = train_error / folds
            # print("Gamma:{}, training_error:{}, validation_error:{}, C:{}".format(gamma, train_error, val_error, c))
            errs.append((train_error, val_error))
            if val_error < lowest_error:
                lowest_error = val_error
                best_gamma = gamma
                best_c = c

    errs = np.array(errs)
    # Já sabemos qual o melhor modelo
    test_error, predictions = svm_test_error(best_gamma, best_c, train_X, train_Y, test_X, test_Y)
    return best_gamma, best_c, predictions, test_error


def gauss_naive_bayes(train_X, train_Y, test_X, test_Y, folds):
    """Returns test error and predictions on the test set with a gaussian Naive Bayes classifier"""
    gauss = GaussianNB()
    gauss.fit(train_X, train_Y)
    predictions = gauss.predict(test_X)
    num_error_test = np.count_nonzero(np.not_equal(predictions, test_Y))
    return predictions, num_error_test / test_Y.shape[0]


def normal_test(n, error_NB, error_SVM, error_gauss):
    stds_NB = np.sqrt(n * error_NB * (1 - error_NB))
    stds_SVM = np.sqrt(n * error_SVM * (1 - error_SVM))
    stds_gauss = np.sqrt(n * error_gauss * (1 - error_gauss))
    x_NB = n * error_NB
    x_SVM = n * error_SVM
    x_gauss = n * error_gauss

    nb_min = x_NB - 1.96 * stds_NB
    nb_max = x_NB + 1.96 * stds_NB
    svm_min = x_SVM - 1.96 * stds_SVM
    svm_max = x_SVM + 1.96 * stds_SVM
    gauss_min = x_gauss - 1.96 * stds_gauss
    gauss_max = x_gauss + 1.96 * stds_gauss
    print("-------Normal test resultados-------")
    print("Intervalo Naive Bayes: {} a {}".format(nb_min, nb_max))
    print("Intervalo SVM: {} a {}".format(svm_min, svm_max))
    print("Intervalo Gaussian Naive Bayes: {} a {}".format(gauss_min, gauss_max))
    compare_normal_intervals(nb_min, nb_max, svm_min, svm_max, "Naive Bayes", "SVM")
    compare_normal_intervals(nb_min, nb_max, gauss_min, gauss_max, "Naive Bayes", "Gaussian Naive Bayes")
    compare_normal_intervals(gauss_min, gauss_max, svm_min, svm_max, "Gaussian Naive Bayes", "SVM")


def compare_normal_intervals(min_1, max_1, min_2, max_2, name_1, name_2):
    if (min_2 <= min_1 <= max_2) or (min_2 <= max_1 <= max_2):
        print("Os classificadores {} e {} não apresentam diferenças significativas, pelo que podem ter performances "
              "semelhantes".format(name_1, name_2))
    elif max_1 < min_2:
        print("O classificador {} aparenta ter melhor performance que o classificador {}".format(name_1, name_2))
    else:
        print("O classificador {} aparenta ter melhor performance que o classificador {}".format(name_2, name_1))


def mcnemar_test(Y, predictions_NB, predictions_SVM, predictions_gauss, error_nb, error_svm, error_gauss):
    print("-------McNemar test resultados-------")
    compare_mcnemar_test(predictions_NB, predictions_SVM, Y, error_nb, error_svm, "Naive Bayes", "SVM")
    compare_mcnemar_test(predictions_NB, predictions_gauss, Y, error_nb, error_gauss, "Naive Bayes",
                         "Gaussian Naive Bayes")
    compare_mcnemar_test(predictions_gauss, predictions_SVM, Y, error_gauss, error_svm, "Gaussian Naive Bayes", "SVM")


def compare_mcnemar_test(predictions_0, predictions_1, Y, error_0, error_1, name_0, name_1):
    e01 = np.count_nonzero(np.logical_and(np.not_equal(Y, predictions_0), np.equal(Y, predictions_1)))
    e10 = np.count_nonzero(np.logical_and(np.not_equal(Y, predictions_1), np.equal(Y, predictions_0)))
    # e10 = np.count_nonzero((Y != predictions_1) and (Y == predictions_0))
    q = ((abs(e01 - e10) - 1) ** 2) / (e01 + e10)
    print("Test McNemar entre {} e {}: {}".format(name_0, name_1, q))
    if q >= 3.84 and error_0 < error_1:
        print("O classificador {} aparenta ter melhor performance que o classificador {}".format(name_0, name_1))
    elif q >= 3.84 and error_1 < error_0:
        print("O classificador {} aparenta ter melhor performance que o classificador {}".format(name_1, name_0))
    else:
        print("Os classificadores {} e {} não apresentam diferenças significativas, pelo que podem ter performances "
              "semelhantes".format(name_0, name_1))


def main():
    folds = 5
    train_X, train_Y, test_X, test_Y = load_data(train_file, test_file)
    best_bandwidth, predictions_NB, true_error_NB = naive_bayes(train_X, train_Y, test_X, test_Y, folds)
    print("Naive Bayes Classifier: Test error: {}, Bandwidth: {}".format(true_error_NB, best_bandwidth))
    best_gamma, predictions_SVM, true_error_SVM = svm_classifier(train_X, train_Y, test_X, test_Y, folds)
    print("SVM Classifier: Test error: {}, Gamma: {}".format(true_error_SVM, best_gamma))
    best_gamma_OPT, best_c, predictions_SVM_OPT, true_error__SVM_OPT = svm_classifier_with_c(train_X, train_Y, test_X,
                                                                                             test_Y, folds)
    print("SVM Classifier with C: Test error: {}, Gamma: {}, C:{}".format(true_error__SVM_OPT, best_gamma_OPT, best_c))
    predictions_GAUSS, true_error_GAUSS = gauss_naive_bayes(train_X, train_Y, test_X, test_Y, folds)
    print("Gaussian Naive Bayes Classifier: Test error: {}".format(true_error_GAUSS))
    normal_test(test_Y.shape[0], true_error_NB, true_error_SVM, true_error_GAUSS)
    mcnemar_test(test_Y, predictions_NB, predictions_SVM, predictions_GAUSS, true_error_NB, true_error_SVM,
                 true_error_GAUSS)
    compare_mcnemar_test(predictions_SVM, predictions_SVM_OPT, test_Y, true_error_SVM, true_error__SVM_OPT, "SVM", "SVM otimizado")


main()
