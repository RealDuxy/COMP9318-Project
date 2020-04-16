# -*- encoding: utf-8 -*-
"""
@File    : helper.py
@Time    : 2020-04-11 03:34
@Author  : Duxy, Chloe
@Software: PyCharm
"""

import pickle

import numpy as np
data_path = './toy_example/example/Data_File_2'
centorid_path = './toy_example/example/Centroids_File_2'
codebooks_path = './toy_example/example/Codebooks_2'
codes_path = './toy_example/example/Codes_2'

def display_data(path):
    '''
    将numpy二进制文件 data_file 可视化
    :return: None
    '''
    with open(path, 'rb') as f:
        Data_File = pickle.load(f, encoding='bytes')
    print(Data_File)
    print(Data_File.shape)


def split_data(data, P=2,axis=1):
    '''
    split data into P parts
    :param data: (N,M)
    :param P:
    :return: (P,N,M/P) A list of sub-arrays.
    '''
    return np.split(data,P,axis)

def L1_distance(point1,point2):
    '''
    L1 distance
    :param point1: (1,M)
    :param point2: (1,M)
    :return: L1 distance: float64
    '''
    # print(point1.shape, point2.shape)
    return np.sum(np.abs(point2 - point1))


# runtest
if __name__ == '__main__':
    display_data(data_path)
    display_data(centorid_path)
    display_data(codebooks_path)
    display_data(codes_path)

    # display_query()
    # split_data()
    # L1_distance()
