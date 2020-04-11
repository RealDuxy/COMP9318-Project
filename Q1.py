# -*- encoding: utf-8 -*-
"""
@File    : Q1.py
@Time    : 2020-04-11 03:50
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

import helper as h


def PQ(data, P, init_centroids, max_iter):
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

    distance  = h.distance(norm=1)


    return codebooks, codes




