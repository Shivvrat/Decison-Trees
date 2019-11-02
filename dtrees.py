import os
import subprocess
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    print("There is a import error we are trying to fix it.")
    print("Please check that you have an internet connection.")
    subprocess.call("pip install " + "numpy")
    subprocess.call("pip install " + "pandas")
    os.startfile(__file__)
    sys.exit()

import dt_entropy_reduced_error_post_pruning
import dt_entropy
import dt_variance_reduced_error_post_pruning
import dt_variance
import dt_variance_depth_pruning
import dt_entropy_depth_pruning
import random_forest

arguments = list(sys.argv)
train_data_name = arguments[1]
valid_data_name = arguments[2]
test_data_name = arguments[3]
type_of_tree = str(arguments[4])
try:
    impurity_measure = str(arguments[5])
    pruning = str(arguments[6])
except IndexError:
    print("You are using random forest and thus don't need to give more than 4 arguments, but if you have not going "
          "to use the random forest algorithm please check the number of arguments provided, they, might be wrong or "
          "even worst less")
    # If we don't give the arguments
    # as in the case of random forests


def import_data():
    d = os.getcwd()
    try:
        train_data = pd.read_csv(os.path.join(d, "{0}".format(train_data_name)), header=None)
        test_data = pd.read_csv(os.path.join(d, "{0}".format(test_data_name)), header=None)
        valid_data = pd.read_csv(os.path.join(d, "{0}".format(valid_data_name)), header=None)
    except:
        print("Either the name/extension of the files are wrong or the files are not in the same folder as the code. "
              "Please check.")
        exit(-1)

    # Now we will return the data frames
    return [test_data, train_data, valid_data]


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


def main():
    test_data, train_data, valid_data = import_data()
    train_y, train_x = get_attributes_and_labels(train_data)
    valid_y, valid_x = get_attributes_and_labels(valid_data)
    test_y, test_x = get_attributes_and_labels(test_data)
    accuracy = 0
    if type_of_tree == "-td":
        if impurity_measure == "-ie":
            if pruning == "-pn":
                accuracy = dt_entropy.main_dtree_with_entropy(train_x, train_y, test_x, test_y)
            elif pruning == "-pd":
                accuracy = dt_entropy_depth_pruning.main_dtree_with_entropy(train_x, train_y, test_x, test_y, valid_x,
                                                                            valid_y)
            elif pruning == "-pr":
                accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y,
                                                                                                       test_x, test_y,
                                                                                                       valid_x, valid_y)
            else:
                print("You have entered wrong command line arguments. Please check.")
                accuracy = -1
        elif impurity_measure == "-iv":
            if pruning == "-pn":
                accuracy = dt_variance.main_dtree_with_variance(train_x, train_y, test_x, test_y)
            elif pruning == "-pd":
                accuracy = dt_variance_depth_pruning.main_dtree_with_variance(train_x, train_y, test_x, test_y, valid_x,
                                                                              valid_y)
            elif pruning == "-pr":
                accuracy = dt_variance_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x,
                                                                                                        train_y,
                                                                                                        test_x, test_y,
                                                                                                        valid_x,
                                                                                                        valid_y)
            else:
                print("You have entered wrong command line arguments. Please check.")
                accuracy = -1
        else:
            print("You have entered wrong command line arguments. Please check.")
            accuracy = -1
    elif type_of_tree == "-tr":
        import warnings
        warnings.filterwarnings("ignore")
        accuracy = random_forest.random_forest(train_x, train_y, test_x, test_y)
    print("The accuracy is :- ", accuracy)


if __name__ == "__main__":
    main()
