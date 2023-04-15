from random import random
from time import sleep
from multiprocessing import Pool


def task(identifier):
    value = random()
    print(f'executing job id: {identifier} with value: {value}')
    sleep(value)
    return value

# if __name__ == '__main__':
with Pool() as pool:
    for result in pool.map(task, range(10)):
        print(f'Got result:{result}')


def preprocess_feature(feature_index_list):
    ## some code to process the features
    return feature_index_list

import multiprocessing
feature_index_list = [[1,2,3], [4,5,6], [7,8,9]]
with multiprocessing.Pool() as pool:
    result = pool.map(preprocess_feature, feature_index_list)
print(result) ## return the preprocessed feature
