import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

"""
This function is used to import the data. Put the data in a folder named all_data in the directory of the code
"""


def import_data(dt_name):
    """

    :param dt_name: Name of the Dataset
    :return: Three pandas frames which correspond to training, testing and validation data
    """
    # First we get the directory of our project and then take all the three files and import them to the respective
    # names.
    d = os.getcwd()
    test_data = pd.read_csv(os.path.join(os.path.join(d, "all_data"), "test_{0}.csv".format(dt_name)), header=None)
    train_data = pd.read_csv(os.path.join(os.path.join(d, "all_data"), "train_{0}.csv".format(dt_name)), header=None)
    validation_data = pd.read_csv(os.path.join(os.path.join(d, "all_data"), "valid_{0}.csv".format(dt_name)),
                                  header=None)
    # Now we will return the data frames
    return [test_data, train_data, validation_data]


"""
This function is defined to get the labels/classes and  attribute values in different variables
"""


def get_attributes_and_labels(data):
    """

    :param data: The dataset to be divided
    :return: Two panda frames which are in order of classes and attributes
    """
    # Here we divide our attributes and classes features for a given dataset
    return [data.iloc[:, -1], data.iloc[:, :-1]]


"""
In this function we try to find the accuracy for our predictions and return the accuracy
"""


def decision_tree_accuracy(test_y, predicted_y):
    """

    :param test_y: The classes given with the data
    :param predicted_y: The predicted classes
    :return: The accuracy for the given tree
    """
    test_y = test_y.tolist()
    right_predict = 0
    # Here we predict the accuracy by the formula accuracy = right_predictions/total_data_points_in_the_set
    for each in range(np.size(test_y)):
        if test_y[each] == predicted_y[each]:
            right_predict = right_predict + 1
    return right_predict / len(test_y)


def random_forest(train_x, train_y, test_x, test_y):
    """

    :param train_x: This is the training data
    :param train_y: This is the training class
    :param test_x:  This is the testing data
    :param test_y: This is the testing class
    :return: The accuracy for created tree
    """
    train_y = pd.DataFrame(train_y)
    max_value_in_target_attribute, temp = np.unique(train_y, return_counts=True)
    max_value_in_target_attribute = max_value_in_target_attribute[0]
    # First we do the training process
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    output_test = clf.predict(test_x)
    # Now we find the accuracy by comparing the predicted output values and the true values
    accuracy = decision_tree_accuracy(test_y, output_test)
    return accuracy
