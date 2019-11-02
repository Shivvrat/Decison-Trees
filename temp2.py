import pickle
import os
import copy
import numpy as np
import pandas as pd
with open('data.p', 'rb') as fp:
    tree = pickle.load(fp)

def change_value_of_sub_tree(tree, new_value, key_value, class_type):
    """

    :param tree: The given tree on which we want to do pruning
    :param new_value: This sis the value we want to put in the tree
    :param key_value: This is the key on which we want to change the value
    :param class_type: This is the class of the node, i.e. 1 or 0
    :return: the tree with new value as the value for key key_value
    """
    if isinstance(tree, dict):
        for key_for_this_tree, next_tree in tree.items():
            if key_for_this_tree == key_value:
                try:
                    tree[key_for_this_tree][class_type] = new_value
                except TypeError:
                    i = 0
                return tree
            else:
                tree = change_value_of_sub_tree(next_tree, new_value, key_value, class_type)
    return tree




"""
This function is used to predict the new and unseen test data point by using the created tree and the given instance
"""


def decision_tree_predict(tree, testing_example, max_value_in_target_attribute):
    """

    :param max_value_in_target_attribute: If we are not able to classify due to less data, we return this value when testing
    :param tree: This is the trained tree which we will use for finding the class of the given instance
    :param testing_example: These are the instance on which we want to find the class
    :return: the predicted value for given datapoint
    """
    predicted_values = []
    # We take each attribute for the datapoint anc check if that attribute is the root node for the tree we are on
    try:
        max_value_in_target_attribute = tree[list(tree)[0]]["majority_target_attribute"]
    except KeyError:
        i = 0
    for each_attribute in list(testing_example.index):
        if each_attribute in tree:
            try:
                # I have used a try catch here since we  trained the algo on a part of the data and it's not
                # necessary that we will have a branch which can classify the data point at all.
                next_dict = tree[each_attribute][testing_example[each_attribute]]
                new_tree = next_dict
            except KeyError:
                # There are two things we can do here, first is to show an error and return no class and the second
                # thing we can do is return the max value in our training target array. error = "The algorithm cannot
                # classify the given instance right now" return error....Need more training
                return max_value_in_target_attribute
            if type(next_dict) == dict:
                # In this case we see if the value predicted is a tree then we again call the recursion if not we
                # return the value we got
                return decision_tree_predict(new_tree, testing_example, max_value_in_target_attribute)
            else:
                return new_tree


"""
This function is used to find the output for the given testing dataset by using the tree created during the training process
"""


def decision_tree_test(test_x, tree, max_value_in_target_attribute):
    """

    :param test_x: This is input attributes from the testing data
    :param tree: This is the tree created after the training process
    :param max_value_in_target_attribute:  This is the most occurring target_attribute in our training data
    :return: The output for the given testing data
    """
    output = []
    # In this function we just use the decision_tree_predict and predict the value for all the instances
    for index, row in test_x.iterrows():
        try:
            output.append(int(decision_tree_predict(tree, row, max_value_in_target_attribute)))
        except TypeError:
            i = 0
    return output


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


[test, train, valid] = import_data("c300_d100")
[train_y, train_x] = get_attributes_and_labels(train)
[test_y, test_x] = get_attributes_and_labels(test)
[valid_y, valid_x] = get_attributes_and_labels(valid)

train_y = pd.DataFrame(train_y)
max_value_in_target_attribute, temp = np.unique(train_y, return_counts=True)
try:
    if temp[0] < temp[1]:
        counts_of_attribute = np.flipud(temp)
        unique_value_of_attribute = np.flipud(max_value_in_target_attribute)
except IndexError:
    i = 0

max_value_in_target_attribute = max_value_in_target_attribute[0]
# First we do the training process
"""
import pickle
with open('data.p', 'wb') as fp:
    pickle.dump(tree, fp, protocol=pickle.HIGHEST_PROTOCOL)
"""

globals()["index"] = 0
def reduced_error_post_pruning(main_tree, given_tree, set_of_pruned_trees, validation_x, validation_y,
                               max_value_in_target_attribute, accuracy_for_main_tree):
    """

    :param main_tree: This is the main tree
    :param set_of_pruned_trees: this is the set we will return which will contain each main tree with pruning which leads to imporved accuracy
    :param max_value_in_target_attribute: This is the maximum occurring value in the train dataset
    :param validation_y: The data given as validation set labels
    :param validation_x: The data given as validation set attributes
    :param given_tree: The tree on which we want to apply the reduced-error post pruning
    :return: The pruned tree
    """
    if isinstance(given_tree, dict):
        current_index = list(given_tree.keys())[0]
    else:
        return given_tree
    try:
        if isinstance(given_tree[current_index][0], dict):
            pruned_0 = reduced_error_post_pruning(main_tree, given_tree[current_index][0], set_of_pruned_trees, validation_x,
                                                  validation_y, max_value_in_target_attribute, accuracy_for_main_tree)
        else:
            return given_tree
    except KeyError:
        i = 1
    try:
        if isinstance(given_tree[current_index][1], dict):
            pruned_1 = reduced_error_post_pruning(main_tree, given_tree[current_index][1], set_of_pruned_trees, validation_x,
                                                  validation_y, max_value_in_target_attribute, accuracy_for_main_tree)
        else:
            return given_tree
    except KeyError:
        i = 1
    copy_of_given_tree = copy.deepcopy(given_tree)
    if isinstance(given_tree, dict):
        given_tree[current_index][0] = given_tree[current_index]["majority_target_attribute"]
        given_tree[current_index][1] = given_tree[current_index]["majority_target_attribute"]
        given_tree[current_index]["type_of_node"] = "leaf"
    else:
        return given_tree
#    predicted_y = decision_tree_test(validation_x, main_tree, max_value_in_target_attribute)
#    accuracy_for_main_tree = decision_tree_accuracy(validation_y, predicted_y)
    copy_of_main_tree = copy.deepcopy(main_tree)
    # Now we change the value for the pruned tree to compare with the main tree validation accuracy
    try:
        change_value_of_sub_tree(copy_of_main_tree, pruned_0, current_index, 0)
    except UnboundLocalError:
        i = 1
    try:
        change_value_of_sub_tree(copy_of_main_tree, pruned_1, current_index, 1)
    except UnboundLocalError:
        i = 0
    predicted_y = decision_tree_test(validation_x, copy_of_main_tree, max_value_in_target_attribute)
    pruned_accuracy = decision_tree_accuracy(validation_y, predicted_y)
    if accuracy_for_main_tree < pruned_accuracy:
        set_of_pruned_trees.update({globals()["index"]: copy_of_main_tree})
        globals()["index"] = globals()["index"] + 1
    return copy_of_given_tree



def choose_best_pruned_tree(main_tree, validation_x, validation_y, max_value_in_target_attribute):
    predicted_y = decision_tree_test(validation_x, main_tree, max_value_in_target_attribute)
    current_best_score = decision_tree_accuracy(validation_y, predicted_y)
    next_best_score = 1
    while next_best_score > current_best_score:
        set_of_pruned_trees = {}
        predicted_y = decision_tree_test(validation_x, main_tree, max_value_in_target_attribute)
        current_best_score = decision_tree_accuracy(validation_y, predicted_y)
        copy_of_main_tree = copy.deepcopy(main_tree)
        reduced_error_post_pruning(copy_of_main_tree, copy_of_main_tree, set_of_pruned_trees, validation_x, validation_y, max_value_in_target_attribute, current_best_score)
        globals()["index"] = 0
        for temp, main_tree_pruned in set_of_pruned_trees.items():
            predicted_y = decision_tree_test(validation_x, main_tree_pruned, max_value_in_target_attribute)
            accuracy_for_pruned_main_tree = decision_tree_accuracy(validation_y, predicted_y)
            if accuracy_for_pruned_main_tree > current_best_score:
                main_tree = main_tree_pruned
                next_best_score = accuracy_for_pruned_main_tree
    return main_tree


pruned_tree = choose_best_pruned_tree(tree, valid_x, valid_y, max_value_in_target_attribute)
output_test = decision_tree_test(test_x, pruned_tree, max_value_in_target_attribute)
# This is the accuracy for the pruned tree
accuracy2 = decision_tree_accuracy(test_y, output_test)
print(accuracy2)
from pprint import pprint
pprint(tree)
pprint(pruned_tree)
