"""
main.py
description: training and evaluating various models
important: explanation on parameters choosing, NN architecture choosing and much more, in the PDF's provided in github:
"https://github.com/eliordadon/ckd-predictor-ml-analysis"
important2: there are sections in the main() functions, you cannot (!) run more than one section at a time,
is it splitted so you can see how to run each one, if you want to run a particular section (for example the Neural
network one), then you need to comment all the other sections.
"""

import logging
from helper import load_data, split_data, split_withval, preprocessing, preprocessing_whole
from models import *
from neural_network import run_eval_nn


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='ckd_predictor.log')
    logging.info('Main Started')

    try:
        # load, EDA (applicable for all kinds of models)
        # load
        data_df = load_data('datasetVariations/CKDF_noQmarks_unindexed.csv')
        logging.info('Data loaded successfully')
        # EDA

        # ==== SECTION 1 - SUPERVISED ML (not deep learning) models ====
        logging.info('Section 1 Started')

        # split (no validation set)
        X_train, X_test, y_train, y_test = split_data(data_df, test_size=0.2)
        logging.info('Data split into training and test sets, no validation')

        # preprocessing data, scaling=discretization
        X_train_p, y_train_p, X_test_p, y_test_p = \
            preprocessing(X_train, y_train, X_test, y_test, scaling_method='discretization')
        logging.info('Data successfully preprocessed')

        # training models
        rf_info = rand_forest(X_train_p, X_test_p, y_train_p)
        cart_info = cart_tree(X_train_p, X_test_p, y_train_p)
        nb_info = naive_bayes(X_train_p, X_test_p, y_train_p)
        logging.info('Models training completed')

        # evaluation of the models
        models = {'Random Forest': rf_info, 'CART': cart_info, 'Naive Baysien': nb_info}
        for model_name, model_info in models.items():
            # if info is None, the model training was not successful, but we won't stop running, just skip
            if model_info is not None:
                # need to add y_test to model info for evaluation
                model_info['y_test'] = y_test_p
                summary = summarize_results(model_name, model_info)
                logging.info(f'Results for model {model_name}: \n {summary}')

        logging.info('Section 1 Ended')
        # ==== End of SECTION 1 ====

        # # ==== SECTION 2 - UNSUPERVISED clustering models (only kmeans for now), no split ====
        # logging.info('Section 2 Started')
        #
        # # separate from target variable as this is an Unsupervised-problem
        # features = data_df.drop('class', axis=1)
        # true_labels = data_df['class']
        #
        # # preprocessing features, scaling=normalization(default)
        # features_p = preprocessing_whole(features)
        # logging.info('Features successfully preprocessed')
        #
        # # training, number of clusters = 2 by default
        # kmeans_labels, centers, data_values = k_means(features_p)
        # logging.info('KMeans model trained successfully')
        #
        # # evaluation
        # logging.info('Starting KMeans model evaluation')
        # k_means_evaluation(data_values, kmeans_labels, centers)
        # k_means_supervised_evaluation(kmeans_labels, true_labels)
        # logging.info('KMeans model evaluated successfully')
        #
        # logging.info('Section 2 Ended')
        # # ==== End of SECTION 2 ====
        #
        # # ==== SECTION 3 - Predefined simple and basic feedforward Neural Network model ====
        # logging.info('Section 3 Started')
        #
        # # split, with validation set
        # X_train, X_val, X_test, y_train, y_val, y_test = split_withval(data_df, test_size=0.2, val_size=0.25)
        # logging.info('Data split into training, validation, and test sets')
        #
        # # preprocessing data, scaling=normalization(default)
        # X_train_p, y_train_p, X_val_p, y_val_p, X_test_p, y_test_p = \
        #     preprocessing(X_train, y_train, X_val, y_val, X_test, y_test)
        # logging.info('Data successfully preprocessed')
        #
        # # get number of features
        # input_dim = X_train_p.shape[1]
        #
        # # train run and evaluate the neural network, ephocs=100, batch size = 32
        # logging.info('Starting NN model training and evaluation')
        # run_eval_nn(X_train_p, y_train_p, X_val_p, y_val_p, X_test_p, y_test_p, input_dim, plot_to_file=False)
        # logging.info('NN model trained and evaluated successfully')
        #
        # logging.info('Section 3 Ended')
        # # ==== End of SECTION 3 ====

    except Exception as e:
        logging.error(f'Encountered an error at main(): {e}')
        raise e

if __name__ == '__main__':
    main()
