"""
helper.py
description: include all the necessary functions that we will use for data handling / preprocessing
important: most functions not used in main() or in other functions here, the reason is the assignment
was in steps, which wasn't a full preprocessing workflow as I implemented after the assignment.
author: Elior Dadon
"""
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


def calc_averages(data):
    """
    calculate the average of instances of numeric attributes in a dataset
    :param data: dataset
    :return: dictionary defined as {attribute_name:average}
    """
    averages = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            averages[col] = data[col].mean()

    return averages


def calc_standard_deviation(data):
    """
        calculate the standard deviation of instances of numeric attributes in a dataset
        :param data: dataset
        :return: dictionary defined as {attribute_name:average}
        """
    stds = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            stds[col] = data[col].std()

    return stds


def has_missing_values(data):
    """
    check for missing values in each column in a database
    :param data:
    :return: a dictionary defined as {attribute_name:True/False} True if there are missing values and False otherwise
    """
    missing = {}
    for col in data.columns:
        if data[col].isnull().values.any():
            missing[col] = True
        else:
            missing[col] = False
    return missing


def remove_dups(data):
    """
    remove duplicate rows
    :param data: dataset
    :return: none
    """
    data.drop_duplicates(inplace=True)


def get_unique_vals(data):
    """
    print unique values for each attribute, mainly used to check for not valid values
    :param data: dataset
    :return: none
    """
    for col in data.columns:
        print(f'{col} has unique values {data[col].unique()}, \n')


def remove_col(data, col_name):
    """
    removes an entire column from the dataset, will not save it to a new file
    :param col_name: column name
    :param data: dataset
    :return: none
    """
    data.drop(col_name, axis=1, inplace=True)


def change_to_binary(data, two_options_nominal_cols):
    """
    replacing all the nominal attributes with two options with binary 0 or 1
    the original dataframe will be modified!
    :param data: dataset (pandas dataframe)
    :param two_options_nominal_cols: dictionary of the form column_name:positive_value, example: rbc:normal
    :return: none
    """
    for col, val in two_options_nominal_cols.items():
        data[col] = data[col].apply(lambda x: 1 if x == val else 0)  # changing to true or false
        data[col] = data[col].astype(int)  # changing it to int 1 or 0


def fill_nominal(data, nominal_cols, fill_method):
    """
    fill nominal attributes missing values in the dataset with the desired method (fill_method - most frequent / etc)
    :param data:
    :param data: dataset (pandas dataframe)
    :param nominal_cols: list of strings where each represent a nominal attribute in data
    :param fill_method: the method to fill the missing value
    :return: none
    """
    imputer = SimpleImputer(strategy=fill_method)
    data[nominal_cols] = imputer.fit_transform(data[nominal_cols])


def fill_numerical(data, numerical_cols, fill_method):
    """
    fill numerical attributes missing values in the dataset with the desired method (fill_method - mean / median / etc)
    :param data: dataset (pandas dataframe)
    :param numerical_cols: list of strings where each represent a numerical attribute in data
    :param fill_method: the method to fill the missing value
    :return: none
    """
    imputer = SimpleImputer(strategy=fill_method)
    # round(2) will limit the after the dot number to 2
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols]).round(2)


def visual_correlation(data):
    """
    create and visualize the correlation matrix between the dataset features
    :param data: pandas dataframe
    :return: none
    """
    corr_mat = data.corr()
    sns.heatmap(corr_mat, cmap='coolwarm')
    plt.show()


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


def data_discretization(data):
    """
    perform discretization on non-binary features using KBinsDiscretizer
    :param data: pandas dataframe represents dataset
    :return: none
    """
    # Choose number of bins
    n_bins = 8

    # Select non-binary numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    nonbinary_numeric_cols = [col for col in numeric_cols if data[col].nunique() > 2]

    # Fit and transform KBinsDiscretizer to non-binary numeric columns
    disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_disc = disc.fit_transform(data[nonbinary_numeric_cols])

    # Replace original non-binary numeric columns with discretized versions
    data[nonbinary_numeric_cols] = X_disc


def normalize_numeric(data):
    """
    Performs normalization to the numeric features to range of [0,1]
    :param data: pandas dataframe represents dataset
    :return: none
    """

    # List of nominal columns
    nominal_cols = ['sg', 'al']

    # Identify binary columns (those with exactly 2 unique values)
    binary_cols = [col for col in data.columns if data[col].nunique() == 2]

    # Determine the remaining numeric columns
    numeric_cols = [col for col in data.columns if col not in binary_cols + nominal_cols]

    # Apply normalization only on the numeric columns (min/max), implemented by hand cuz of uni (could use MinMaxScaler)
    for col in numeric_cols:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    # controlling the number of decimal points, it is in comments because I don't want this piece of code
    # to save to a new file each time. it is preferably to make this chunk in the main function.
    # just use round(decimal=2) (2 decimal numbers or different)


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


def preprocessing(X_train, X_test, y_train, y_test, scaling_method='normalization'):
    """
    Perform preprocessing steps for machine learning. Validation set not included.
    important: the function now implemented with prints along the way for understanding what happened.
    it's not mandatory and the print statements can be removed.
    :param X_train: training features
    :param X_test: test features
    :param y_train: training target
    :param y_test: test target
    :param scaling_method: scaling method ('discretization' or 'normalization' only for now)
    :return: preprocessed X_train, X_test, y_train, y_test (numpy arrays!)
    """
    # concatenate X and y for train and test sets
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    print("Training data shape:", train.shape)
    print("Test data shape:", test.shape)

    # remove duplicates
    remove_dups(train)
    remove_dups(test)
    print("After removing duplicates - Training data shape:", train.shape)
    print("After removing duplicates - Test data shape:", test.shape)

    # fill nominal and numerical missing values
    nominal_cols = train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print("Nominal columns:", nominal_cols)
    print("Numerical columns:", numerical_cols)

    nominal_imputer = SimpleImputer(strategy='most_frequent')
    numerical_imputer = SimpleImputer(strategy='mean')

    train[nominal_cols] = pd.DataFrame(
        nominal_imputer.fit_transform(train[nominal_cols]),
        columns=nominal_cols,
        index=train.index
    )
    print("Missing values in nominal columns after imputation - Train: ", train[nominal_cols].isnull().sum().sum())

    train[numerical_cols] = pd.DataFrame(
        numerical_imputer.fit_transform(train[numerical_cols]),
        columns=numerical_cols,
        index=train.index
    )
    print("Missing values in numerical columns after imputation - Train: ", train[numerical_cols].isnull().sum().sum())

    test[nominal_cols] = pd.DataFrame(
        nominal_imputer.transform(test[nominal_cols]),
        columns=nominal_cols,
        index=test.index
    )
    print("Missing values in nominal columns after imputation - Test: ", test[nominal_cols].isnull().sum().sum())

    test[numerical_cols] = pd.DataFrame(
        numerical_imputer.transform(test[numerical_cols]),
        columns=numerical_cols,
        index=test.index
    )
    print("Missing values in numerical columns after imputation - Test: ", test[numerical_cols].isnull().sum().sum())

    # change to binary
    two_options_nominal_cols = {col: train[col].mode()[0] for col in nominal_cols if train[col].nunique() == 2}
    change_to_binary(train, two_options_nominal_cols)
    change_to_binary(test, two_options_nominal_cols)
    print("After binary transformation - Training data:\n", train.head())
    print("After binary transformation - Test data:\n", test.head())

    # define feature and target columns
    feature_cols = [col for col in train.columns if col != 'class']
    target_col = 'class'

    # scaling
    if scaling_method == 'discretization':
        disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
        train[feature_cols] = disc.fit_transform(train[feature_cols])
        test[feature_cols] = disc.transform(test[feature_cols])
    elif scaling_method == 'normalization':
        scaler = MinMaxScaler()
        train[feature_cols] = scaler.fit_transform(train[feature_cols])
        test[feature_cols] = scaler.transform(test[feature_cols])

    # feature selection
    features_to_keep = feature_selection(train)
    train = train[features_to_keep]
    test = test[features_to_keep]
    print("After feature selection - Training data shape:", train.shape)
    print("After feature selection - Test data shape:", test.shape)
    print("Features kept:", features_to_keep)

    # separating X and y
    y_train = train.pop(target_col)
    X_train = train
    y_test = test.pop(target_col)
    X_test = test

    return X_train, X_test, y_train, y_test


def preprocessing_withval(X_train, X_val, X_test, y_train, y_val, y_test, scaling_method='normalization'):
    """
    Perform preprocessing steps for machine learning. Validation included.
    important: the function now implemented with prints along the way for understanding what happened.
    it's not mandatory and the print statements can be removed.
    :param X_train: training features
    :param X_val: validation features
    :param X_test: test features
    :param y_train: training target
    :param y_val: validation target
    :param y_test: test target
    :param scaling_method: scaling method ('discretization' or 'normalization' only for now)
    :return: preprocessed X_train, X_test, y_train, y_test (numpy arrays!)
    """
    # concatenate X and y for train, validation and test sets
    train = pd.concat([X_train, y_train], axis=1)
    validation = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # remove duplicates
    remove_dups(train)
    remove_dups(validation)
    remove_dups(test)

    # fill nominal and numerical missing values
    nominal_cols = train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    nominal_imputer = SimpleImputer(strategy='most_frequent')
    numerical_imputer = SimpleImputer(strategy='mean')

    train[nominal_cols] = pd.DataFrame(
        nominal_imputer.fit_transform(train[nominal_cols]),
        columns=nominal_cols,
        index=train.index
    )

    train[numerical_cols] = pd.DataFrame(
        numerical_imputer.fit_transform(train[numerical_cols]),
        columns=numerical_cols,
        index=train.index
    )

    validation[nominal_cols] = pd.DataFrame(
        nominal_imputer.transform(validation[nominal_cols]),
        columns=nominal_cols,
        index=validation.index
    )

    validation[numerical_cols] = pd.DataFrame(
        numerical_imputer.transform(validation[numerical_cols]),
        columns=numerical_cols,
        index=validation.index
    )

    test[nominal_cols] = pd.DataFrame(
        nominal_imputer.transform(test[nominal_cols]),
        columns=nominal_cols,
        index=test.index
    )

    test[numerical_cols] = pd.DataFrame(
        numerical_imputer.transform(test[numerical_cols]),
        columns=numerical_cols,
        index=test.index
    )

    # change to binary
    two_options_nominal_cols = {col: train[col].mode()[0] for col in nominal_cols if train[col].nunique() == 2}
    change_to_binary(train, two_options_nominal_cols)
    change_to_binary(validation, two_options_nominal_cols)
    change_to_binary(test, two_options_nominal_cols)

    # define feature and target columns
    feature_cols = [col for col in train.columns if col != 'class']
    target_col = 'class'

    # scaling
    if scaling_method == 'discretization':
        disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
        train[feature_cols] = disc.fit_transform(train[feature_cols])
        validation[feature_cols] = disc.transform(validation[feature_cols])
        test[feature_cols] = disc.transform(test[feature_cols])
    elif scaling_method == 'normalization':
        scaler = MinMaxScaler()
        train[feature_cols] = scaler.fit_transform(train[feature_cols])
        validation[feature_cols] = scaler.transform(validation[feature_cols])
        test[feature_cols] = scaler.transform(test[feature_cols])

    # feature selection
    features_to_keep = feature_selection(train)
    train = train[features_to_keep]
    validation = validation[features_to_keep]
    test = test[features_to_keep]
    print("After feature selection - Training data shape:", train.shape)
    print("Features kept:", features_to_keep)

    # separating X and y
    y_train = train.pop(target_col)
    X_train = train
    y_val = validation.pop(target_col)
    X_val = validation
    y_test = test.pop(target_col)
    X_test = test

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocessing_whole(data, scaling_method='normalization'):
    """
    Perform preprocessing steps on whole data, mainly for unsupervised tasks.
    :param data: pandas dataframe
    :param scaling_method: scaling method ('discretization' or 'normalization' only for now)
    :return: preprocessed data (numpy array!)
    """
    # remove duplicates
    remove_dups(data)
    print("Data shape after removing duplicates: ", data.shape)

    # fill nominal and numerical missing values
    nominal_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print("Nominal columns: ", nominal_cols)
    print("Numerical columns: ", numerical_cols)

    nominal_imputer = SimpleImputer(strategy='most_frequent')
    numerical_imputer = SimpleImputer(strategy='mean')

    data[nominal_cols] = pd.DataFrame(
        nominal_imputer.fit_transform(data[nominal_cols]),
        columns=nominal_cols,
        index=data.index
    )

    data[numerical_cols] = pd.DataFrame(
        numerical_imputer.fit_transform(data[numerical_cols]),
        columns=numerical_cols,
        index=data.index
    )
    print("Data after filling missing values: ")
    print(data.head())

    # change to binary
    two_options_nominal_cols = {col: data[col].mode()[0] for col in nominal_cols if data[col].nunique() == 2}
    change_to_binary(data, two_options_nominal_cols)
    print("Data after binary transformation: ")
    print(data.head())

    # scaling
    if scaling_method == 'discretization':
        disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
        data = disc.fit_transform(data)
    elif scaling_method == 'normalization':
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

    return data