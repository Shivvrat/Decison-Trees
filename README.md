# Decison-Trees




## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project
In this homework I have implemented the decision tree learning algorithm. 
1. Implement the decision tree learning algorithm. The main step in this algorithm is choosing the next attribute to split on. Implement the following two heuristics for selecting the next attribute :-
    * Information gain heuristic
    * Variance impurity heuristic
2. Implement the reduced-error post pruning algorithm 
3. Implement depth-based pruning by using maximum depth dmax as a hyper-parameter, namely in your decision tree prune all nodes having depth larger than dmax. We will assume that dmax takes values from the following set: {5,10,15,20,50,100}.

The following algorithms will be compared :-
1. Naive Decision tree learner with Entropy as the impurity heuristic – Naive Decision tree learner with Variance as the impurity heuristic.
2. Decision tree learner with Entropy as the impurity heuristic and reduced error pruning
3. Decision tree learner with Variance as the impurity heuristic and reduced error pruning
4. Decision tree learner with Entropy as the impurity heuristic and depth-based pruning
5. Decision tree learner with Variance as the impurity heuristic and depth-based pruning
6. Random Forests



### Built With

* [Python 3.7](https://www.python.org/downloads/release/python-370/)


## Getting Started

Lets see how to run this program on a local machine.

### Prerequisites

You will need the following modules 
```
import numpy as np 
import pandas as pd 
import os
import subprocess 
import sys
import copy as copy
```
### Installation

1. Clone the repo
```sh
git clone https://github.com/Shivvrat/Decison-Trees.git
```
Use the main.py to run the algorithm.


<!-- USAGE EXAMPLES -->
## Usage
Please enter the following command line argument:-
```sh
python dtrees.py [name_of_train_data_file] [name_of_validation_data_file] [name_of_test_data_file] [type_of_tree_to_use] [impurity_measure] [ pruning]
```
Please use the following command line parameters for the main.py file :-
* Name of train data file
Provide the name of training data file with extension (please provide .csv files only)
* Name of validation data 
 file Provide the name of validation data file with extension (please provide .csv files only)
* Name of test data file 
 Provide the name of testing data file with extension (please provide .csv files only) 
(All the above files should be in the same folder as the code.)

Please use all these functionalities(mentioned below) as in or the code will give an error of wrong command line arguments.
* Type of tree to use 
    * -td for dtrees
* impurity measure 
    * ie 
     for entropy 
    * iv 
     for variance
* pruning 
    * -pn 
    For no pruning 
    * -pd 
    For depth based pruning 
    * -pr 
    For reduced error pruning
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - Shivvrat Arya[@ShivvratA](https://twitter.com/ShivvratA) - shivvratvarya@gmail.com

Project Link: [https://github.com/Shivvrat/Decison-Trees.git](https://github.com/Shivvrat/Decison-Trees.git)
