import os

import numpy as np
import pandas as pd

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
This function is used to find the entropy which is our impurity heuristic for this algorithm
"""


def get_entropy(data):
    """

    :param data: THese are the values for which we want to find the entropy of. We pass a whole vector of values which
    correspond to the attribute of importance and find entropy for that vector.
    :return: Entropy for the given vector
    """
    entropy_value = 0
    temp, unique_count = np.unique(data, return_counts=True)
    # We will use the formula mentioned in the slides to calculate the value of entropy for both the options (i.e,
    # 1 and 0)
    sum_of_counts = np.sum(unique_count)
    for count in unique_count:
        entropy_value = entropy_value - ((count / sum_of_counts) * np.log2(count / sum_of_counts))
    return entropy_value


"""
This function is used to find the information gain for the given sub-tree/tree. The information gain is used to find the
attribute we will use to do further branching
"""


def Information_Gain_Heuristic(examples, attributes, target_attribute):
    """

    :param examples: The data for whihc we want to find the information gain
    :param attributes: the values of the attributes available (the column number)
    :param target_attribute: the target attribute we are trying to find
    :return: Information Gain of the given sub-tree.
    """
    # Here we find the entropy for the root node
    previous_entropy = get_entropy(target_attribute)
    Information_Gain = []
    for each_attribute in attributes:
        unique_value_of_attribute, counts_of_attribute = np.unique(examples[each_attribute], return_counts=True)
        try:
            if counts_of_attribute[0] < counts_of_attribute[1]:
                counts_of_attribute = np.flipud(counts_of_attribute)
                unique_value_of_attribute = np.flipud(unique_value_of_attribute)
        except IndexError:
            i = 0
        # Since I have hardcoded the array_after_division arrays we will try to the first values for 0.
        if unique_value_of_attribute[0] == 1:
            counts_of_attribute = np.flipud(counts_of_attribute)
            unique_value_of_attribute = np.flipud(unique_value_of_attribute)
        array_after_division_1 = []
        array_after_division_0 = []
        # This loop is for 0 and 1
        # I need to find the number of 1's and 0's in target value when the given attribute value is something
        # particular
        total_data = pd.concat([examples, target_attribute], axis=1, sort=False)
        # Here I concatenated the data frames so that only one df is used to both lookup the value and find the value
        # to append
        row_names = total_data.index.values
        list_of_row_names = list(row_names)
        for each in list_of_row_names:
            value_to_append = int(total_data.iloc[:, -1][each])
            if examples[each_attribute][each] == 1:
                array_after_division_1.append(value_to_append)
            else:
                array_after_division_0.append(value_to_append)
        # Here I will use try catch since if the target_attribute have only one unique value then it will give an
        # error if we try to use the second index (i.e. 2). and if we have only one unique value then our imputrity
        # is 0 and thus entropy is 0
        try:
            value_of_new_inpurity = (counts_of_attribute[0] / np.size(examples[each_attribute])) * get_entropy(
                array_after_division_0) + (counts_of_attribute[1] / np.size(examples[each_attribute])) * get_entropy(
                array_after_division_1)
        except IndexError:
            value_of_new_inpurity = 0
        temp = previous_entropy - value_of_new_inpurity
        Information_Gain.append(temp)
    return Information_Gain


"""
This function is the main function for our algorithm. The decision_tree function is used recursively to create new nodes
and make the tree while doing the training.
"""


def decision_tree_construction(examples, target_attribute, attributes, depth):
    """

    :param examples: The data we will use to train the tree(x)
    :param target_attribute: The label we want to classify(y)
    :param attributes: The number(index) of the labels/attributes of the data-set
    :return: The tree corresponding to the given data
    """
    # This is the first base condition of the algorithm. It is used if the attributes variable is empty, then we return
    # the single-node tree Root, with label = most common value of target_attribute in examples
    # The base condition for the recursion when we check if all the variables are same or not in the node and if they
    # are same then we return that value as the node
    if len(attributes) == 0 or len(np.unique(target_attribute)) == 1:
        unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
        try:
            if counts_of_attribute[0] < counts_of_attribute[1]:
                unique_value_of_attribute = np.flipud(unique_value_of_attribute)
        except IndexError:
            i = 0
        if unique_value_of_attribute[0] == 1:
            # More positive values
            return 1, depth
        elif unique_value_of_attribute[0] == 0:
            # More negative values
            return 0, depth
    # This is the recursion part of the algorithm in which we try to find the sub-tree's by using recursion and
    # information gain
    else:
        Information_Gain = Information_Gain_Heuristic(examples, attributes, target_attribute)
        best_attribute_number = attributes[np.argmax(Information_Gain)]
        # Since we now have the best_attribute(A in algorithm) we will create the root node of the tree/sub-tree with
        # that and name the root as the best attribute among all Here we make the tree as a dictionary for testing
        # purposes
        tree = dict([(best_attribute_number, dict())])
        if isinstance(tree, int):
            # If the given value is a int value then it's definitely a leaf node and if it's a dictionary then its a
            # node
            tree[best_attribute_number]["type_of_node"] = "leaf"
            tree[best_attribute_number]["depth"] = depth
            unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
            try:
                if counts_of_attribute[0] < counts_of_attribute[1]:
                    counts_of_attribute = np.flipud(counts_of_attribute)
                    unique_value_of_attribute = np.flipud(unique_value_of_attribute)
            except IndexError:
                # Here we can have an index error since in some case it may happen that the array has only one type
                # of value and thus accessing the index [1] is not possible
                i = 0
            tree[best_attribute_number]["majority_target_attribute"] = unique_value_of_attribute[0]
            tree[best_attribute_number]["best_attribute_number"] = best_attribute_number
        else:
            tree[best_attribute_number]["type_of_node"] = "node"
            tree[best_attribute_number]["depth"] = depth
            unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
            try:
                if counts_of_attribute[0] < counts_of_attribute[1]:
                    counts_of_attribute = np.flipud(counts_of_attribute)
                    unique_value_of_attribute = np.flipud(unique_value_of_attribute)
            except IndexError:
                # Here we can have an index error since in some case it may happen that the array has only one type
                # of value and thus accessing the index [1] is not possible
                i = 0
            tree[best_attribute_number]["majority_target_attribute"] = unique_value_of_attribute[0]
            tree[best_attribute_number]["best_attribute_number"] = best_attribute_number

        attributes.remove(best_attribute_number)
        # Now we do the recursive algorithm which will be used to create the tree after the root node.
        depth_of_node = []
        for each_unique_value in np.unique(examples[best_attribute_number]):
            # We use those values for which the examples[best_attribute_number] == each_unique_value
            class1 = each_unique_value
            new_target_attribute = pd.DataFrame(target_attribute)
            total_data = pd.concat([examples, new_target_attribute], axis=1, sort=False)
            # WE do this step so that we can pick the values which belong to the best_attribute = [0,1], i.e. We now
            # want to divide our data so that the values for the best_attribute is divided among the branches. And
            # thus we will have 4 arrays now, two for the data and two for target attribute.
            new_data_after_partition = total_data.loc[total_data[best_attribute_number] == class1]
            new_target_attribute, new_examples_after_partition = get_attributes_and_labels(new_data_after_partition)
            # This is also a condition for our algorithm in which we check if the number of examples after the
            # partition are positive or not. If the values are less than 1 then we return the most frequent value in
            # the node
            if len(new_examples_after_partition) == 0:
                unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
                try:
                    if counts_of_attribute[0] < counts_of_attribute[1]:
                        counts_of_attribute = np.flipud(counts_of_attribute)
                        unique_value_of_attribute = np.flipud(unique_value_of_attribute)
                except IndexError:
                    i = 0
                if unique_value_of_attribute[0] == 1:
                    # More positive values
                    return 1, depth
                elif unique_value_of_attribute[0] == 0:
                    # More negative values
                    return 0, depth
            # This is the recursion step, in which we make new decision trees till the case when any of the base
            # cases are true
            new_sub_tree_after_partition, deptha = decision_tree_construction(new_examples_after_partition,
                                                                              new_target_attribute, attributes,
                                                                              depth + 1)
            depth_of_node.append(deptha)
            # Here we are adding the depth of the node so that we can do the depth based pruning
            tree[best_attribute_number][each_unique_value] = new_sub_tree_after_partition
            if isinstance(new_sub_tree_after_partition, int):
                tree[best_attribute_number]["type_of_node"] = "leaf"
                tree[best_attribute_number]["depth"] = depth
                unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
                try:
                    if counts_of_attribute[0] < counts_of_attribute[1]:
                        counts_of_attribute = np.flipud(counts_of_attribute)
                        unique_value_of_attribute = np.flipud(unique_value_of_attribute)
                except IndexError:
                    i = 0
                tree[best_attribute_number]["majority_target_attribute"] = unique_value_of_attribute[0]
                tree[best_attribute_number]["best_attribute_number"] = best_attribute_number
            else:
                tree[best_attribute_number]["type_of_node"] = "node"
                tree[best_attribute_number]["depth"] = depth

                unique_value_of_attribute, counts_of_attribute = np.unique(target_attribute, return_counts=True)
                try:
                    if counts_of_attribute[0] < counts_of_attribute[1]:
                        counts_of_attribute = np.flipud(counts_of_attribute)
                        unique_value_of_attribute = np.flipud(unique_value_of_attribute)
                except IndexError:
                    i = 0
                tree[best_attribute_number]["majority_target_attribute"] = unique_value_of_attribute[0]
                tree[best_attribute_number]["best_attribute_number"] = best_attribute_number
    return tree, max(depth_of_node)


"""
This function is used to do the pruning of the given tree based on the given max_depth
"""


def depth_pruning_for_one_value(given_tree, maximum_allowed_depth, current_depth=0):
    """

    :param given_tree: This is the tree we want to prune based on the depth
    :param maximum_allowed_depth: This is the maximum allowed depth for the main tree
    :param current_depth: This is the current depth we are on
    :return: The depth pruned tree
    """
    for each_key in list(given_tree):
        # In this function we are just checking if the depth is greater or not and if greater we are
        # pruning the tree
        if isinstance(given_tree[each_key], dict):
            try:
                current_depth = given_tree[each_key]["depth"]
            except KeyError:
                # Here we are not anything to the depth since in this case the node will be a leaf.
                current_depth = current_depth
            if current_depth == maximum_allowed_depth:
                try:
                    given_tree[each_key] = given_tree[each_key]["majority_target_attribute"]
                except KeyError:
                    given_tree[each_key] = 0
            else:
                depth_pruning_for_one_value(given_tree[each_key], maximum_allowed_depth, current_depth)


"""
This function is used for the depth based pruning for validation of the values
"""


def depth_pruning_by_validation_set(given_tree, valid_x, valid_y, max_value_in_target_attribute):
    """

    :param given_tree: This is the tree we want to prune based on the depth
    :param valid_x: This is the validation data
    :param valid_y: This is the validation class
    :param max_value_in_target_attribute:  This is the max value in target attribute (for testing purposes)
    :return:
    """
    list = [5, 10, 15, 20, 50, 100]
    best_accuracy = 0
    best_number = 0
    for each in list:
        #  Here we just iterate over the values and try to find the best hyper parameter of depth
        pruned_tree = given_tree.copy()
        depth_pruning_for_one_value(pruned_tree, each, 0)
        predicted_y = decision_tree_test(valid_x, pruned_tree, max_value_in_target_attribute)
        accuracy_for_pruned_tree = decision_tree_accuracy(valid_y, predicted_y)
        if accuracy_for_pruned_tree > best_accuracy:
            best_accuracy = accuracy_for_pruned_tree
            best_number = each
    return best_accuracy, best_number


"""
This function is used to predict the new and unseen test data point by using the created tree and the given instance
"""


def decision_tree_predict(tree, testing_example, max_value_in_target_attribute):
    """

    :param max_value_in_target_attribute: If we are not able to classify due to less data, we return this value when testing
    :param tree: This is the trained tree which we will use for finding the class of the given instance
    :param testing_example: These are the instance on which we want to find the class
    :return:
    """
    # We take each attribute for the datapoint anc check if that attribute is the root node for the tree we are on
    try:
        max_value_in_target_attribute = tree[list(tree)[0]]["majority_target_attribute"]
    except (KeyError, IndexError):
        max_value_in_target_attribute = 0
    for each_attribute in list(testing_example.index):
        if each_attribute in tree:
            try:
                value_of_the_attribute = testing_example[each_attribute]
                # I have used a try catch here since we  trained the algo on a part of the data and it's not
                # necessary that we will have a branch which can classify the data point at all.
                new_tree = tree[each_attribute][value_of_the_attribute]
            except KeyError:
                # There are two things we can do here, first is to show an error and return no class and the second
                # thing we can do is return the max value in our training target array. error = "The algorithm cannot
                # classify the given instance right now" return error....Need more training
                return max_value_in_target_attribute
            except IndexError:
                # This is the case when we do pruning and the node becomes a value.
                return tree[each_attribute]
            if type(new_tree) == dict:
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
        output.append(int(decision_tree_predict(tree, row, max_value_in_target_attribute)))
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


"""
In this function we will try to iterate over the whole naive decision tree with entropy as the impurity heuristic
"""


def main_dtree_with_entropy(train_x, train_y, test_x, test_y, valid_x, valid_y):
    """

    :param train_x: This is the training data
    :param train_y: This is the training class
    :param test_x:  This is the testing data
    :param test_y: This is the testing class
    :return: The accuracy for created tree
    """
    train_y = pd.DataFrame(train_y)
    max_value_in_target_attribute, temp = np.unique(train_y, return_counts=True)
    try:
        if temp[0] < temp[1]:
            counts_of_attribute = np.flipud(temp)
            unique_value_of_attribute = np.flipud(max_value_in_target_attribute)
    except IndexError:
        i = 0
    max_value_in_target_attribute = max_value_in_target_attribute[0]
    train_y = pd.DataFrame(train_y)
    # First we do the training process
    tree, depth = decision_tree_construction(train_x, train_y, list(train_x.columns), 0)
    # Then we do the pruning of the tree and choose best attribute for depth and return the accuracy
    accuracy, best_number = depth_pruning_by_validation_set(tree, valid_x, valid_y, max_value_in_target_attribute)
    return accuracy
