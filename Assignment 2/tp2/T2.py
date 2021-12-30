import numpy as np

import tp2_aux as aux
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from tempfile import TemporaryFile
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd

labels_src = r"C:\Users\joana\Documents\Escola\Engenharia Informática\4ºAno\1ºSemestre\Aprendizagem Automática\ProjetosAA\Assignment 2\tp2\labels.txt"


def save_file(filename, data):
    outfile = open(filename, 'wb')
    np.savetxt(outfile, data, delimiter=',')


def pca(data):
    pca = PCA(n_components=6)
    pca.fit(data)
    t_data = pca.transform(data)
    # print(t_data)

    return t_data


def tsne(data):
    tsne_estimator = TSNE(n_components=6, method="exact")
    t_data = tsne_estimator.fit_transform(data)
    # print(t_data)
    return t_data


def isomap(data):
    isomap_estimator = Isomap(n_components=6)
    t_data = isomap_estimator.fit_transform(data)
    # print(t_data)
    return t_data


def feature_extraction(data):
    data_pca = pca(data)
    data_tsne = tsne(data)
    new_data = np.append(data_pca, data_tsne, axis=1)
    data_isomap = isomap(data)
    new_data = np.append(new_data, data_isomap, axis=1)
    return new_data


def standardize_data(data):
    mean = np.mean(data, 0)
    std = np.std(data, 0)
    standerized_data = (data - mean) / std
    return standerized_data


def features_selected_names(features_selected):
    names = []
    for i in features_selected:
        if i < 6:
            names.append(f"{i}-pca{i}")
        elif 6 <= i < 12:
            names.append(f"{i}-tsne{i%6}")
        else:
            names.append(f"{i}-isomap{i%12}")
    return names

def feature_selection(data):
    labels = np.loadtxt(labels_src, delimiter=",")
    Y = labels[labels[:, 1] != 0, -1]
    X = data[labels[:, 1] != 0, :]
    selector = SelectKBest(f_classif, k=6)
    labeled_data = selector.fit_transform(X, Y)
    selected_features = selector.get_support()
    selected_features_indexes = selector.get_support(indices=True)
    f, prob = f_classif(labeled_data, Y)
    new_data = data[:, selected_features]

    names = features_selected_names(selected_features_indexes)

    df = pd.DataFrame(data=new_data, columns=names)

    print(names)
    print(f)
    scatter_matrix(df, alpha=0.5, figsize=(15, 10), diagonal='kde')
    plt.savefig("scatter_matrix.png")
    """
    ['0-pca0', '1-pca1', '2-pca2', '9-tsne3', '12-isomap0', '13-isomap1']
    [ 7.84258393 46.63388152 19.18778367 13.67991868 21.99504424 42.17749338]
    
    Vamos selecionar [1,2,4,5]
    """
    # plt.show()
    plt.close()

    save_file("features_selected", new_data)

    return new_data


#       DBSCAN      |
def find_eps(data):
    k_neighbours_classifier = KNeighborsClassifier(n_neighbors=4)
    k_neighbours_classifier = k_neighbours_classifier.fit(data, np.zeros(data.shape[0]))

    dist, ixs = k_neighbours_classifier.kneighbors(data)
    distk = np.sort(dist[:, -1])

    return distk[3]


def perform_DBSCAN(data, features_ids):
    eps = find_eps(data)

    dbscan = DBSCAN(eps=eps, min_samples=4).fit(data)
    dbscan_lbs = dbscan.labels_

    aux.report_clusters(features_ids, dbscan_lbs, f"{data.columns.values} dbscan_report.html")
    return dbscan_lbs


def main(are_features_selected):
    global features_selected

    if are_features_selected:
        features_selected_6 = np.loadtxt("features_selected", delimiter=',')
        # select = [1,2,4,5]
        selected = np.array([False, True, True, False, True, True])
        features_selected = features_selected_6[:,selected]


    else:
        matrix_image = aux.images_as_matrix() / 255.0
        features_18 = feature_extraction(matrix_image)
        # Standardize features
        features_standardized = standardize_data(features_18)
        features_selected = feature_selection(features_standardized)


main(True)
