"""
helper.py
description: include all the necessary functions that we will use for data handling / preprocessing
author: Elior Dadon
"""
import logging
import pandas as pd
from sklearn.impute import SimpleImputer  # filling missing values
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler  # discretization, normalization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # split to train/test set


def replace_question_marks(csv_file):
    """
    *this is specific for the purpose of converting the questions marks in arff format to a csv compatible format*
    replace any "?" in an csv file with nan values (missing values)
    :param csv_file: csv file to operate on
    :return: none
    """
    # load CSV file
    df = pd.read_csv(csv_file, delimiter=',', na_values='?')

    # write to new CSV file
    df.to_csv("CKDF.csv", index=False)


def remove_dups(data_list):
    """
    remove duplicate rows on each dataset from data_list (inplace)
    :param data_list: list of pandas dataframes
    :return: None
    """
    for i, data in enumerate(data_list):
        logging.info(f'Data shape before removing duplicates in dataframe {i + 1}: {data.shape}')
        data.drop_duplicates(inplace=True)
        logging.info(f'Data shape after removing duplicates in dataframe {i + 1}: {data.shape}')


def fill_missing_values(data_list, nominal_cols, numerical_cols, nominal_strategy, numerical_strategy):
    """
    fill missing values in nominal / numerical columns based on strategies (inplace)
    :param data_list: list of pandas dataframes
    :param nominal_cols: list of nominal columns
    :param numerical_cols: list of numerical columns
    :param nominal_strategy: filling method to apply for nominal columns
    :param numerical_strategy: filling method to apply for numeric columns
    :return: None
    """
    nominal_imputer = SimpleImputer(strategy=nominal_strategy)
    numerical_imputer = SimpleImputer(strategy=numerical_strategy)

    for i, df in enumerate(data_list):
        logging.info(f"Missing values before imputation in dataframe {i + 1}: {df.isna().sum().sum()}")

        # If it's the first dataframe (training set), fit and transform
        if i == 0:
            df[nominal_cols] = pd.DataFrame(
                nominal_imputer.fit_transform(df[nominal_cols]),
                columns=nominal_cols,
                index=df.index
            )

            df[numerical_cols] = pd.DataFrame(
                numerical_imputer.fit_transform(df[numerical_cols]),
                columns=numerical_cols,
                index=df.index
            )
        # Otherwise, only transform (validation / test)
        else:
            df[nominal_cols] = pd.DataFrame(
                nominal_imputer.transform(df[nominal_cols]),
                columns=nominal_cols,
                index=df.index
            )

            df[numerical_cols] = pd.DataFrame(
                numerical_imputer.transform(df[numerical_cols]),
                columns=numerical_cols,
                index=df.index
            )

        logging.info(f"Missing values after imputation in dataframe {i + 1}: {df.isna().sum().sum()}")


def change_to_binary(data_list, two_options_nominal_cols):
    """
    replacing all the nominal attributes with two options with binary 0 or 1
    the original dataframe will be modified.
    :param data_list: list of datasets (pandas dataframes)
    :param two_options_nominal_cols: dictionary of the form column_name:positive_value, example: rbc:normal
    :return: None
    """
    for i, data in enumerate(data_list):
        columns_changed = []  # list to store columns which were changed
        for col, val in two_options_nominal_cols.items():
            data[col] = data[col].apply(lambda x: 1 if x == val else 0)  # changing to true or false
            data[col] = data[col].astype(int)  # changing it to int 1 or 0
            columns_changed.append(col)

        logging.info(f"Columns '{', '.join(columns_changed)}' were changed to binary in dataframe number {i + 1}.")


def scale_values(data_list, feature_cols, scaling_method):
    """
    performs scaling on the data provided, the scaling_method will determine how the scaling will be performed
    important: the order of the datasets are important, it assuming TRAIN-VALIDATION-TEST or TRAIN-TEST or just whole
    data, if the order is different, the behaviour will be unwanted, because we want to fit_transform on the training
    set and use the parameters on the other sets.
    :param data_list: list of datasets (pandas dataframes)
    :param feature_cols: columns without the target
    :param scaling_method: determine which scaling method to use (normalization / discretization only for now)
    :return: True if scaling went successfully, False otherwise
    """
    # Create the scaler outside of the loop.
    if scaling_method == 'discretization':
        scaler = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
    elif scaling_method == 'normalization':
        scaler = MinMaxScaler()
    else:
        logging.warning('Scaling method provided not implemented / undefined')
        return False

    for i, data in enumerate(data_list):
        try:
            if i == 0:  # training data, fit_transform (or whole data if not splitted)
                data[feature_cols] = scaler.fit_transform(data[feature_cols])
            else:  # only transform, using the training data parameters (validation / test)
                data[feature_cols] = scaler.transform(data[feature_cols])
        except (ValueError, TypeError) as e:
            logging.error(f'Error occurred during scaling (helper.scale_values function): {e}')
            return False

    return True


def feature_selection(data):
    """
    performs feature selection by removing features based on the correlation matrix; 
    highly correlated features with other independent variable (not the target variable) will be removed;
    the threshold for removing a feature will be 0.6;
    features which are highly correlated with the target variable will not be removed even if they are
    highly correlated with other independent variables
    :param data: pandas dataframe 
    :return: none
    """

    # Set the threshold for correlation coefficient
    threshold = 0.6

    # Create the correlation matrix
    corr_matrix = data.corr()

    # Create a set to store the names of the features to remove
    features_to_remove = set()

    # Loop through the correlation matrix to find highly correlated features with other independent variables
    for i in range(len(corr_matrix.columns)):
        # loop through the upper triangle of the correlation matrix,
        # excluding the diagonal (to not make duplicate calculations)
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                feature_i = corr_matrix.columns[i]
                feature_j = corr_matrix.columns[j]
                if feature_i != 'class' and feature_j != 'class':
                    features_to_remove.add(feature_i)
                    features_to_remove.add(feature_j)

    # Check the correlation between the features in the features_to_remove set with the target variable
    # for strongly correlated features with the target variable, we want to keep them
    features_to_keep = set()
    for feature in features_to_remove:
        if abs(corr_matrix.loc[feature, 'class']) > threshold:
            features_to_keep.add(feature)

    # Calculate final set of features to keep
    features_to_keep = set(data.columns) - (features_to_remove - features_to_keep)

    return list(features_to_keep)


def separate_target(data, target_col='class'):
    """
    separate the features from the target variable
    :param data: dataset, pandas dataframe
    :param target_col: string, represent the target column name
    :return: X (features), y (target)
    """
    y = data.pop(target_col)
    X = data
    return X, y


def visual_correlation(data):
    """
    create and visualize the correlation matrix between the dataset features
    :param data: pandas dataframe
    :return: none
    """
    corr_mat = data.corr()
    sns.heatmap(corr_mat, cmap='coolwarm')
    plt.show()


def data_histograms(data):
    """
    create histogram for each of the dataset features
    :param data: pandas dataframe
    :return: none
    """

    fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(15, 15))

    for i, ax in enumerate(axs.flatten()):
        ax.hist(data.iloc[:, i], bins=20)
        ax.set_title(data.columns[i])

    plt.tight_layout()
    plt.show()


def load_data(data_path):
    """
    load the data into a pandas dataframe
    :param data_path: dataset path
    :return: the dataset as pandas dataframe
    """
    return pd.read_csv(data_path)


def split_data(data, test_size=0.2):
    """
    split the data into training set and test set, validation set isn't implemented here.
    :param data: pandas dataframe
    :param test_size: test size portion, default is 20%
    :return: training set, and test set
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=test_size,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def split_withval(data, test_size=0.2, val_size=0.25):
    """
    split the data into training set, validation set, and test set
    :param data: pandas dataframe
    :param test_size: test size portion, default is 20%
    :param val_size: validation size portion, default is 20% (0.25 will give 60/20/20 split)
    :return: training set, validation set and test set
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=test_size,
                                                        random_state=42)

    # further split into validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocessing(X_train, y_train, X_test, y_test, X_val=None, y_val=None, scaling_method='normalization'):
    """
    Perform preprocessing steps for machine learning. Validation set is optional.
    :param X_train: training features
    :param y_train: training target
    :param X_test: test features
    :param y_test: test target
    :param X_val: validation features
    :param y_val: validation target
    :param scaling_method: scaling method ('discretization' or 'normalization' only for now)
    :return: preprocessed provided sets (numpy arrays!)
    """
    logging.info('Starting data preprocessing')
    try:
        # concatenate X and y for train and test sets
        train = pd.concat([X_train, y_train], axis=1)
        validation = None if X_val is None else pd.concat([X_val, y_val], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        if validation is None:
            data_list = [train, test]
        else:
            data_list = [train, validation, test]

        # remove duplicates
        logging.info(f'Removing duplicates activated')
        remove_dups(data_list)

        # fill nominal and numerical missing values
        nominal_cols = train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        logging.info('Starting imputation process, filling missing values. columns are:')
        logging.info(f'nominal: {nominal_cols}')
        logging.info(f'numerical: {numerical_cols}')
        fill_missing_values(data_list, nominal_cols, numerical_cols,
                            nominal_strategy='most_frequent', numerical_strategy='mean')

        # change to binary
        logging.info('Starting binary transformation')
        two_options_nominal_cols = {col: train[col].mode()[0] for col in nominal_cols if train[col].nunique() == 2}
        change_to_binary(data_list, two_options_nominal_cols)
        logging.info(f'After binary transformation - training data:')
        logging.info(f'{train.head()}')

        # define feature and target columns
        feature_cols = [col for col in train.columns if col != 'class']
        target_col = 'class'

        # scaling
        if scale_values(data_list, feature_cols, scaling_method=scaling_method):
            logging.info(f'Data scaling performed successfully, method used: {scaling_method}')

        # unpack from data_list
        if validation is None:
            train, test = data_list
        else:
            train, validation, test = data_list

        # feature selection
        logging.info('Starting feature selection')
        features_to_keep = feature_selection(train)
        train = train[features_to_keep]
        if validation is not None:
            validation = validation[features_to_keep]
        test = test[features_to_keep]
        logging.info(f'Feature selection performed successfully, features kept: {features_to_keep}')

        # separating X and y
        X_train, y_train = separate_target(train, target_col)
        X_test, y_test = separate_target(test, target_col)
        if validation is not None:
            X_val, y_val = separate_target(validation, target_col)

        if validation is None:
            return X_train, y_train, X_test, y_test
        else:
            return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logging.error("Exception occurred at helper.preprocessing_whole function: ", exc_info=True)
        raise e


def preprocessing_whole(data, scaling_method='normalization'):
    """
    Perform preprocessing steps on whole data, mainly for unsupervised tasks.
    important - assuming data come with no target column
    :param data: pandas dataframe
    :param scaling_method: scaling method ('discretization' or 'normalization' only for now)
    :return: preprocessed data (numpy array!)
    """
    logging.info('Starting preprocessing whole data')
    try:
        data_list = [data]

        # remove duplicates
        logging.info(f'Removing duplicates activated')
        remove_dups(data_list)

        # fill nominal and numerical missing values
        nominal_cols = data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        logging.info('Starting imputation process, filling missing values. columns are: \n')
        logging.info(f'nominal: {nominal_cols}')
        logging.info(f'numerical: {numerical_cols}')
        fill_missing_values(data_list, nominal_cols, numerical_cols,
                            nominal_strategy='most_frequent', numerical_strategy='mean')

        # change to binary
        logging.info('Starting binary transformation')
        two_options_nominal_cols = {col: data[col].mode()[0] for col in nominal_cols if data[col].nunique() == 2}
        change_to_binary(data_list, two_options_nominal_cols)

        # scaling
        if scale_values(data_list, data, scaling_method=scaling_method):
            logging.info(f'Data scaling performed successfully, method used: {scaling_method}')

        return data

    except Exception as e:
        logging.error("Exception occurred at helper.preprocessing_whole function: ", exc_info=True)
        raise e
