def PackageInstall(error):
    import time, subprocess, os, sys
    lib = str(error)[15:].replace('\'', '')
    print('>>>', str(error))
    print('>>> Download will start after five seconds')
    time.sleep(5)
    subprocess.call("pip install " + lib)
    print('>>> Restarting')
    os.startfile(__file__)
    sys.exit()


try:
    import os
    from pprint import pprint
    import numpy as np
    import pandas as pd

except ImportError as error:
    PackageInstall(error)

import random_forest
"""
[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c300_d100")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c300_d100 ->" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c300_d1000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c300_d1000" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c300_d5000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c300_d5000" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c500_d100")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c500_d100" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c500_d1000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c500_d1000" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c500_d5000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c500_d5000" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1000_d100")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1000_d100" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1000_d1000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1000_d1000" + str(accuracy))
[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1000_d5000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1000_d5000" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1500_d100")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1500_d100" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1500_d1000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1500_d1000" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1500_d5000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1500_d5000" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1800_d100")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1800_d100" + str(accuracy))

[test, train, valid] = dt_entropy_reduced_error_post_pruning.import_data("c1800_d1000")
[train_y, train_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(train)
[test_y, test_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(test)
[valid_y, valid_x] = dt_entropy_reduced_error_post_pruning.get_attributes_and_labels(valid)
accuracy = dt_entropy_reduced_error_post_pruning.main_dtree_with_reduced_error_pruning(train_x, train_y, test_x, test_y, valid_x, valid_y)
pprint("c1800_d1000" + str(accuracy))
"""
[test, train, valid] = random_forest.import_data("c1800_d5000")
[train_y, train_x] = random_forest.get_attributes_and_labels(train)
[test_y, test_x] = random_forest.get_attributes_and_labels(test)
[valid_y, valid_x] = random_forest.get_attributes_and_labels(valid)
accuracy = random_forest.random_forest(train_x, train_y, test_x, test_y)
pprint("c1800_d5000 " + str(accuracy))
