# -*- encoding: utf-8 -*-
"""
@File    : helper.py
@Time    : 2020-04-11 03:34
@Author  : Duxy, Chloe
@Software: PyCharm
"""
import submission
import pickle
import time

def distance(norm=1):
    '''
    计算L1 distance
    :param norm: L-norm distance
    :return: distance
    '''
    distance = 1
    return distance

def display_centroids():
    '''
    将numpy二进制文件 centroids_file 可视化
    :return: None
    '''
    with open('./toy_example/Centroids_File', 'rb') as f:
        Centroids_File = pickle.load(f, encoding='bytes')
    print(Centroids_File)

def display_data():
    '''
    将numpy二进制文件 data_file 可视化
    :return: None
    '''
    with open('./toy_example/Data_File', 'rb') as f:
        Data_File = pickle.load(f, encoding='bytes')
    print(Data_File)

def display_query():
    '''
    将numpy二进制文件 query_file 可视化
    :return: None
    '''
    with open('./toy_example/Query_File', 'rb') as f:
        Query_File = pickle.load(f, encoding='bytes')
    print(Query_File)

# runtest
if __name__ == '__main__':
    display_centroids()
    display_data()
    display_query()
