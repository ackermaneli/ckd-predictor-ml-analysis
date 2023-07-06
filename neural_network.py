"""
neural_network.py
description: include all the necessary functions that we will use for neural network
             model training, prediction and evaluating
author: Elior Dadon
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def build_model(input_dim):
    """
    Build the neural network model.
    :param input_dim: the number of input features
    :return: the neural network model
    """
    # model architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(9, input_dim=input_dim, activation='relu'))  # hidden layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # output layer

    # compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def run_network(train_data, train_labels, val_data, val_labels, model, epochs=100, batch_size=32):
    """
    Train and evaluate the network.
    :param train_data: the training data
    :param train_labels: the training labels
    :param val_data: the validation data
    :param val_labels: the validation labels
    :param model: the neural network model
    :param epochs: the number of training epochs
    :param batch_size: the batch size for training
    :return: the history of the model training
    """
    # Train the network
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_labels))

    return history


def plot_training_history(history):
    """
    Plot the training history of the neural network.
    :param history: the history object obtained from model training
    :return: None
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(14, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(14,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_nn(nn_model, file_name):
    """
    plots the neural network model into a file
    :param nn_model: the neural network model
    :param file_name: file name which the plotting will be exported to
    :return: none
    """
    tf.keras.utils.plot_model(nn_model, to_file=file_name, show_shapes=True)
    print(nn_model.summary())


def report_misclassifications(model, test_data, test_labels):
    """
    Report instances where the model made a wrong prediction.
    :param model: the trained model
    :param test_data: the test data
    :param test_labels: the true labels of the test data
    :return: None
    """
    print("== Misclassified report ==")

    # Get the model's predictions
    y_pred = model.predict(test_data)
    y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to class labels

    # Find where the predictions do not match the true labels
    misclassified = np.where(test_labels != y_pred)

    # Print the misclassified instances
    for index in misclassified[0]:
        print(f"Instance index {index} was misclassified:")
        print(test_data.iloc[index])
        print(f"True label = {test_labels.iloc[index]}, Predicted label = {y_pred[index]}\n")

    # Print confusion matrix for additional insight
    cm = confusion_matrix(test_labels, y_pred)
    print("Confusion Matrix:")
    print(cm)


def run_eval_nn(train_data, train_labels, val_data, val_labels, test_data, test_labels, input_dim, plot_to_file=False):
    """
    Combine all the necessary steps for running the neural network model for the required section.
    :param train_data: the training data
    :param train_labels: the training labels
    :param val_data: the validation data
    :param val_labels: the validation labels
    :param test_data: the test data
    :param test_labels: the test labels
    :param input_dim: number of features not included the target variable
    :return: none
    """

    # build the network
    model = build_model(input_dim)

    # train the network
    history = run_network(train_data, train_labels, val_data, val_labels, model)

    # evaluate the model on the test data
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
    print('\nTest accuracy', test_acc)

    # report misclassified instances
    report_misclassifications(model, test_data, test_labels)

    if plot_to_file:
        # plot the model
        plot_nn(model, 'nn_model.png', plot_to_file)

    # plot the training history as a function of the epochs
    plot_training_history(history)

