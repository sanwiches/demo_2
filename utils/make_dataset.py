import random

class Make_dataset:
    def __init__(self, divide_method):
        self.divide_method = divide_method
        
    def problem_set(self, train_or_test):

        if self.divide_method == "CEC2013LSGO":
            train_problem_set = [1, 4, 5, 8, 9, 12, 13]
            test_problem_set = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        elif self.divide_method == "BNS":
            train_problem_set = [1,2,3,4,5,6,7,8,9,10,11,12]
            test_problem_set = [1,2,3,4,5,6,7,8,9,10,11,12]

        if train_or_test == "train":
            return train_problem_set
        elif train_or_test == "test":
            return test_problem_set
    

