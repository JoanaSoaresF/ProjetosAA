import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from pandas.plotting import scatter_matrix
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

import tp2_aux as aux

labels_src = r"C:\Users\joana\Documents\Escola\Engenharia Informática\4ºAno\1ºSemestre\Aprendizagem Automática\ProjetosAA\Assignment 2\tp2\labels.txt"



def save_file(filename, data):
    outfile = open(filename, 'wb')
    np.savetxt(outfile, data, delimiter=',')
    outfile.close()


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
            names.append(f"{i}-tsne{i % 6}")
        else:
            names.append(f"{i}-isomap{i % 12}")
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
    ['0-pca0', '1-pca1', '2-pca2', '7-tsne1', '12-isomap0', '13-isomap1']
    [ 7.84258393 46.63388161 19.18778378  7.99139872 21.99504424 42.17749338]
    
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


#   Evaluators
def precision(tp, fp):
    result = 0
    if tp + fp != 0:
        result = (tp / (tp + fp))
    return result


def recall(tp, fn):
    result = 0
    if tp + fn != 0:
        result = (tp / (tp + fn))
    return result


def rand(n, tp, tn):
    return ((tp + tn) / (n * (n - 1) / 2))


def f1(precision, recall):
    result = 0
    if precision + recall != 0:
        result = 2 * ((precision * recall) / (precision + recall))
    return result


def confusion_matrix(true_labels, prediction_labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(true_labels)):
        for j in range(i, len(true_labels)):
            if true_labels[i] == true_labels[j] and prediction_labels[i] == prediction_labels[j]:
                tp += 1
            elif true_labels[i] != true_labels[j] and prediction_labels[i] != prediction_labels[j]:
                tn += 1
            elif true_labels[i] != true_labels[j] and prediction_labels[i] == prediction_labels[j]:
                fp += 1
            elif true_labels[i] == true_labels[j] and prediction_labels[i] != prediction_labels[j]:
                fn += 1

    return tp, tn, fp, fn


def performance_measures(data, result_labels):
    """internal index, the silhouette score, and external indexes computed from the labels available: the Rand index,
    Precision, Recall, the F1 measure and the adjusted Rand index. """
    global silhouette
    n_labels = np.unique(result_labels).shape[0]
    if n_labels > 1:
        silhouette = silhouette_score(data, result_labels)
        print(f"Silhouette Score:{silhouette} ")
    else:
        print("Cannot compute Silhouette Score because there is only one cluster")

    true_labels = labels[labels[:, 1] != 0, -1]
    prediction_labels = result_labels[labels[:, 1] != 0]

    ari = adjusted_rand_score(labels_true=true_labels, labels_pred=prediction_labels)
    print(f"Adjusted Rand Score:{ari} ")

    tp, tn, fp, fn = confusion_matrix(true_labels, prediction_labels)
    rand_score = rand(true_labels.shape[0], tp, tn)
    print(f"Rand Score:{rand_score} ")

    precision_val = precision(tp, fp)
    print(f"Precision:{precision_val} ")

    recall_val = recall(tp, fn)
    print(f"Recall:{recall_val} ")

    f1_val = f1(precision_val, recall_val)
    print(f"F1:{f1_val} ")

    return silhouette, ari, rand_score, precision_val, recall_val, f1_val


#       DBSCAN      |
# def find_eps(data):
#     k_neighbours_classifier = KNeighborsClassifier(n_neighbors=4)
#     k_neighbours_classifier = k_neighbours_classifier.fit(data, np.zeros(data.shape[0]))
#
#     dist, ixs = k_neighbours_classifier.kneighbors(data)
#     distk = np.sort(dist[:, -1])
#
#     return distk[3]


def find_range_eps(data):
    k_neighbours_classifier = KNeighborsClassifier(n_neighbors=4)
    k_neighbours_classifier = k_neighbours_classifier.fit(data, np.zeros(data.shape[0]))

    dist, ixs = k_neighbours_classifier.kneighbors(data)
    distk = np.sort(dist[:, -1])[::-1]

    # plots
    plt.plot(ids, distk)
    plt.xlabel("points sorted by dist")
    plt.ylabel("distances")
    plt.title("elbow curve")
    plt.savefig("DBSCAN/neighboursIDS_distance.png")
    plt.close()

    elbow = round(float(KneeLocator(ids, distk, curve='convex', direction='decreasing').knee_y), 2)
    print(f"Knee of curve = {elbow}")

    return elbow


def perform_DBSCAN(data, eps):
    # eps = find_eps(data)
    dbscan = DBSCAN(eps=eps).fit(data)
    dbscan_lbs = dbscan.labels_

    aux.report_clusters(features_ids, dbscan_lbs, f"{data.columns.values} dbscan_report.html")
    return dbscan_lbs


#       K-Means      |
def perform_KMeans(data, k):
    kmeans = KMeans(n_clusters=k).fit(data)
    labels = kmeans.labels_
    silhouette, ari, rand_score, precision_val, recall_val, f1_val = performance_measures(data, labels)
    kmeans_performance.append((silhouette, ari, rand_score, precision_val, recall_val, f1_val))
    aux.report_clusters(ids, labels, f"K-Means/k-means k={k}.html")
    return labels


def perform_affinity_propagation(data, d):
    affinity = AffinityPropagation(damping=d, random_state=0)
    affinity.fit(data)
    labels = affinity.labels_
    silhouette, ari, rand_score, precision_val, recall_val, f1_val = performance_measures(data, labels)
    affinity_propagation_performance.append((silhouette, ari, rand_score, precision_val, recall_val, f1_val))
    aux.report_clusters(ids, labels, f"Affinity Propagation/Affinity_Propagation_damping={d}.html")


def main(are_features_selected):
    global features_selected

    if are_features_selected:
        features_selected_6 = np.loadtxt("features_selected", delimiter=',')
        # select = [1,2,4,5]
        selected = np.array([False, True, True, False, True, True])
        features_selected = features_selected_6[:, selected]
    else:
        matrix_image = aux.images_as_matrix() / 255.0
        features_18 = feature_extraction(matrix_image)
        # Standardize features
        features_standardized = standardize_data(features_18)
        features_selected = feature_selection(features_standardized)

    # eps = find_eps(features_selected)
    # print(f"Calculated eps: {eps}")
    # for e in np.arange(eps, eps + 0.1, 0.02):
    #     print(f"\n\nDBSCAN with eps={e}")
    #     perform_DBSCAN(features_selected, e)
    eps = find_range_eps(features_selected)
    print(f"Calculated eps: {eps}")
    for e in np.arange(eps * 0.85, eps * 1.15, 0.02):
        print(f"\n\nDBSCAN with eps={e}")
        perform_DBSCAN(features_selected, e)

    for k in range(2, 7):
        print(f"\n\nK-Means with k={k}")
        perform_KMeans(features_selected, k)

    for d in np.arange(0.5, 1, 0.05):
        print(f"\n\nAffinity Propagation with damping={d}")
        perform_affinity_propagation(features_selected, d)

    plot(dbscan_performance, "Epsilon Distance", "", "DBSCAN Performance", range(63, 87, 2))
    print(f"\n\nDBSCAN with eps={eps}")
    dbscan_labels = perform_DBSCAN(features_selected, eps)
    aux.report_clusters(ids, dbscan_labels, f"Chosen DBSCAN eps={eps}.html")

    plot(kmeans_performance, "Number of Clusters", "", "K-Means Performance", range(2, 7))
    print(f"\n\nK-Means with k={3}")
    kmeans_labels = perform_KMeans(features_selected, 3)
    aux.report_clusters(ids, kmeans_labels, f"Chosen k-means k={3}.html")

    plot(affinity_propagation_performance, "Damping (x100)", "", "Affinity Propagation Performance", range(50, 100, 5))


def plot(measures, x_label, y_label, graphic_title, x_values):
    """
    Plots the graphic
   silhouette_scores, aris, rand, precision, recall, f1
    """
    m = np.array(measures)

    fig = plt.figure(figsize=(8, 8), frameon=True)
    plt.plot(x_values, m[:, 0], '-b', linewidth=3, label="Silhouette")
    plt.plot(x_values, m[:, 1], '-r', linewidth=3, label="Adjusted Rand Score")
    plt.plot(x_values, m[:, 2], '-g', linewidth=3, label="Rand Score")
    plt.plot(x_values, m[:, 3], '-c', linewidth=3, label="Precision")
    plt.plot(x_values, m[:, 4], '-m', linewidth=3, label="Recall")
    plt.plot(x_values, m[:, 5], '-y', linewidth=3, label="F1")
    plt.title(graphic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper right")
    file_name = ""
    if "K" in graphic_title:
        file_name = "K-Means_performance"
    elif "Affinity" in graphic_title:
        file_name = "Affinity_Propagation_performance"
    else:
        file_name = "DBSCAN_performance"
    plt.savefig("../{}.png".format(file_name), dpi=300)
    plt.show()
    plt.close()


main(True)
