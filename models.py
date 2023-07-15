"""
models.py
description: include all the necessary functions that we will use for model training, prediction and evaluating
author: Elior Dadon
"""
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB  # for Naive-Bayes model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # evaluating clustering K-Means
import numpy as np
from scipy.spatial.distance import cdist, pdist  # for clustering evaluation
import pandas as pd


def rand_forest(X_train, X_test, y_train):
    """
    splits the dataset (data) into training and test sets
    create a Random Forest model and trains it on the training set
    then makes prediction on the test set
    :param X_train: training features
    :param X_test: test features
    :param y_train: training target
    :return: a dictionary with all the necessary stuff for further evaluating:
    {'model': model, 'y_pred': y_pred, 'y_proba': y_proba}
    """
    try:
        # train a Random Forest model with default hyperparameters
        rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X_train, y_train)
        # make predictions on the test set
        y_pred = rf.predict(X_test)
        # calculate predicted probabilities for each class
        y_proba = rf.predict_proba(X_test)

        rf_dict_info = {'model': rf, 'y_pred': y_pred, 'y_proba': y_proba}
        return rf_dict_info
    except Exception as e:
        logging.warning(f'Random Forest model training failed: {e}')
        return None


def cart_tree(X_train, X_test, y_train):
    """
    same for the Random forest one just with the CART algorithm (default in DecisionTreeClassifier)
    we chosen max_depth=3 for not making overfitting and for easy interpretations
    :param X_train: training features
    :param X_test: test features
    :param y_train: training target
    :return: a dictionary with all the necessary stuff for further evaluating:
    {'model': model, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'y_pred': y_pred}
    """
    try:
        cart_model = DecisionTreeClassifier(max_depth=3)
        cart_model.fit(X_train, y_train)
        y_pred = cart_model.predict(X_test)
        y_proba = cart_model.predict_proba(X_test)

        cart_dict_info = {'model': cart_model, 'y_pred': y_pred, 'y_proba': y_proba}
        return cart_dict_info
    except Exception as e:
        logging.warning(f'CART model training failed: {e}')
        return None


def naive_bayes(X_train, X_test, y_train):
    """
    this function trains a Naive Bayes model using the Gaussian Naive Bayes algorithm on the given training set and
    then applies the trained model to the test set to make predictions.
    :param X_train: training features
    :param X_test: test features
    :param y_train: training target
    :return:
    """
    try:
        # initialize Naive Bayes model
        nb = GaussianNB()
        # fit the model
        nb.fit(X_train, y_train)
        # make predictions on test set
        y_pred = nb.predict(X_test)
        # calculate predicted probabilities for each class
        y_proba = nb.predict_proba(X_test)

        nb_dict_info = {'model': nb, 'y_pred': y_pred, 'y_proba': y_proba}
        return nb_dict_info
    except Exception as e:
        logging.warning(f'Naive bayes model training failed: {e}')
        return None


def k_means(data, n_clusters=2):
    """
    :param data: input data, no labels
    :param n_clusters: the number of clusters to form
    :return: cluster labels, cluster centers and the input data
    """
    try:
        # if data is a DataFrame and needs to be converted to a numpy array
        if isinstance(data, pd.DataFrame):
            data_values = data.values
        else:
            data_values = data

        # initialize KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # fit the data
        kmeans.fit(data_values)
        # get the cluster labels and cluster centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        return labels, centers, data_values
    except Exception as e:
        logging.warning(f'KMeans model training failed: {e}')
        raise e


def k_means_evaluation(data, labels, centers, n_clusters=2):
    """

    :param data: input data
    :param labels: cluster labels
    :param centers: cluster centers
    :param n_clusters: the number of clusters formed by kmeans
    :return: None
    """
    # calculate intra-cluster distance (average distance within clusters)
    intra_cluster_distance = np.sum(np.min(cdist(data, centers, 'euclidean'), axis=1)) / data.shape[0]

    # calculate inter-cluster distance (minimum distance between clusters)
    inter_cluster_distance = np.min(pdist(centers, 'euclidean'))

    # calculate silhouette score
    silhouette_avg = silhouette_score(data, labels)

    logging.info(f" ==== Clustering regular evaluation ==== ")
    logging.info(f"For n_clusters = {n_clusters}, average intra-cluster distance is : {intra_cluster_distance}")
    logging.info(f"For n_clusters = {n_clusters}, minimum inter-cluster distance is : {inter_cluster_distance}")
    logging.info(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

    # Visualize the clusters and centroids
    plt.figure(figsize=(8, 5))

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')

    plt.title(f"Visualization of clustered data", fontweight='bold')
    plt.show()


def k_means_supervised_evaluation(kmeans_labels, true_labels):
    """

    :param kmeans_labels:
    :param true_labels: actual labels present in the dataset, assuming no conversion made to binary 0 or 1
    :return: None
    """
    true_labels = (true_labels == 'ckd').astype(int)

    accuracy = accuracy_score(true_labels, kmeans_labels)
    conf_mat = confusion_matrix(true_labels, kmeans_labels)

    logging.info(f"==== Clustering supervised evaluation ==== ")
    logging.info(f'Accuracy: {accuracy}')
    logging.info('Confusion matrix:')
    logging.info(conf_mat)


def evaluate_accuracy(y_test, y_pred, model_name):
    """
    evaluating a model (name 'model_name') and printing the accuracy of it on the test set y_test
    :param model_name: the model name
    :param y_test: test set
    :param y_pred: prediction
    :return: none
    """
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'{model_name} accuracy: {accuracy}')


def cl_report(y_test, y_pred, model_name):
    """
    printing the classification report of a given -model_name- model
    :param y_test: test set
    :param y_pred: prediction
    :param model_name: the model name
    :return: none
    """
    logging.info(f'{model_name} classification report: \n {classification_report(y_test, y_pred)}')


def plot_roc(model_info, model_name):
    """
    plot the ROC curve of a given model on a given dataset and the roc accuracy AUC
    :param model_name:  the model name
    :param model_info: dictionary with all the necessary stuff for the roc curve process
    :return: none
    """
    # Plot the ROC curve for the Random Forest model
    fpr, tpr, _ = roc_curve(model_info['y_test'], model_info['y_proba'][:, 1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve - ' + model_name)
    plt.show()


def confusion_mat(y_test, y_pred, model_name):
    """
    print the confusion matrix
    :param y_test:
    :param y_pred:
    :param model_name:
    :return:
    """
    # Assuming having trained and made predictions using a model and have the y_test and y_pred variables available
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f'{model_name} Confusion matrix: \n {cm}')


def summarize_results(model_name, model_info):
    """
    Generate a summary report with evaluation metrics for a given model and dataset
    important: models usually not include y_test in the model_info dictionary, thus it is necessary to add it
    before calling this function.
    :param model_info: a dictionary with the necessary model information
    :param model_name: the name of the model
    :return: a string with the summary report
    """
    y_test = model_info['y_test']
    y_pred = model_info['y_pred']
    y_proba = model_info['y_proba']
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba[:, 1])

    summary = f"\n=== {model_name} ===\n\n"
    summary += f"Confusion Matrix:\n{cm}\n\n"
    summary += f"Classification Report:\n{report}\n"
    summary += f"Accuracy: {accuracy:.4f}\n"
    summary += f"Weighted Kappa: {cohen_kappa_score(y_test, y_pred, weights='linear'):.4f}\n"
    summary += f"AUC: {auc_score:.4f}\n\n"

    return summary
