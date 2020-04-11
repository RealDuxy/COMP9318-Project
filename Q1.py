# -*- encoding: utf-8 -*-
"""
@File    : Q1.py
@Time    : 2020-04-11 03:50
@Author  : Duxy Chole
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

import helper as h
import pickle
import time

def pq(data, P, init_centroids, max_iter):
    '''
    PQ method working with L1 distance

    :param data: an array with shape (N,M) and dtype='float32'
           where N is the number of vectors and M is the dimensionality.
    :param P: the number of partitions/blocks the vector will be split into.
           P >= 2, default M
    :param init_centroids: an array with shape (P,K,M/P) and dtype='float32'
    :param max_iter: maximum number of iterations
    :return: codebooks: an array with shape (P, K, M/P) and dtype='float32'
             codes: an array with shape (N, P) and dtype=='uint8'
    '''
    codebooks = 1
    codes = 1
    distance  = h.distance(norm=1)
    print(distance)


    return codebooks, codes

# runtest for Q1
if __name__ == '__main__':
    with open('./toy_example/Data_File', 'rb') as f:
        Data_File = pickle.load(f, encoding='bytes')
    with open('./toy_example/Centroids_File', 'rb') as f:
        Centroids_File = pickle.load(f, encoding='bytes')
    start = time.time()
    codebooks, codes = pq(data=Data_File, P=2, init_centroids=Centroids_File, max_iter=20)
    end = time.time()
    time_cost_1 = end - start
    print(f'Runtime: {time_cost_1}')







