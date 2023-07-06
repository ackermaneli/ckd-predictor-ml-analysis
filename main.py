"""
main.py
description: training and evaluating various models
important: explanation on parameters choosing, NN architecture choosing and much more, in the PDF's provided in github:
"https://github.com/eliordadon/ckd-predictor-ml-analysis"
important2: there are sections in the main() functions, you cannot (!) run more than one section at a time,
is it splitted so you can see how to run each one, if you want to run a particular section (for example the Neural
network one), then you need to comment all the other sections.
"""
from helper import load_data, split_data, split_withval, preprocessing, preprocessing_withval, preprocessing_whole
from models import *
from neural_network import run_eval_nn


def main():
    # load, EDA (applicable for all kinds of models)
    data_df = load_data('datasetVariations/CKDF_noQmarks_unindexed.csv')
    # EDA (didn't implemented here)

    # ==== SECTION 1 - SUPERVISED ML (not deep learning) models ====
    # split (no validation set)
    X_train, X_test, y_train, y_test = split_data(data_df, test_size=0.2)
    # preprocessing data, scaling=discretization
    X_train_p, X_test_p, y_train_p, y_test_p = \
        preprocessing(X_train, X_test, y_train, y_test, scaling_method='discretization')
    # training models
    rf_info = rand_forest(X_train_p, X_test_p, y_train_p)
    cart_info = cart_tree(X_train_p, X_test_p, y_train_p)
    nb_info = naive_bayes(X_train_p, X_test_p, y_train_p)
    # evaluation of the models
    models = {'Random Forest': rf_info, 'CART': cart_info, 'Naive Baysien': nb_info}
    for model_name, model_info in models.items():
        # need to add y_test to model info for evaluation
        model_info['y_test'] = y_test_p
        summary = summarize_results(model_name, model_info)
        print(summary)

    # # ==== SECTION 2 - UNSUPERVISED clustering models (only kmeans for now), no split ====
    # # separate from target variable as this is an Unsupervised problem
    # features = data_df.drop('class', axis=1)
    # true_labels = data_df['class']
    # # preprocessing features, scaling=normalization(default)
    # features_p = preprocessing_whole(features)
    # # training, number of clusters = 2 by default
    # kmeans_labels, centers, data_values = k_means(features)
    # # evaluation
    # k_means_evaluation(data_values, kmeans_labels, centers)
    # k_means_supervised_evaluation(kmeans_labels, true_labels)

    # # ==== SECTION 3 - Predefined simple and basic feedforward Neural Network model ====
    # # split, with validation set
    # X_train, X_val, X_test, y_train, y_val, y_test = split_withval(data_df, test_size=0.2, val_size=0.25)
    # # preprocessing data, scaling=normalization(default)
    # X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p = \
    #     preprocessing_withval(X_train, X_val, X_test, y_train, y_val, y_test)
    # # get number of features
    # input_dim = X_train_p.shape[1]
    # # train run and evaluate the neural network, ephocs=100, batch size = 32
    # run_eval_nn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p, y_test_p, input_dim, plot_to_file=False)


if __name__ == '__main__':
    main()
