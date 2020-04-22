# -*- encoding: utf-8 -*-
"""
@File    : Q1.py
@Time    : 2020-04-11 03:50
@Author  : Duxy Chole
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import numpy as np
import pickle
import time

data_path = './example/Data_File_2'
centorid_path = './example/Centroids_File_2'
codebooks_path = './example/Codebooks_2'
codes_path = './example/Codes_2'
p = 4


def pq(data, P, init_centroids, max_iter=20):
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
    data_blocks = split_data(data,P)  # 得到有p个array的list
    codebooks = []
    code = []
    for i in range(P):
        centroids = init_centroids[i]
        data_block = data_blocks[i]
        epoch = 1
        while epoch <= max_iter:
            print('epoch: ',epoch,'      ',max_iter)
            center = K_means(centroids, data_block)
            epoch += 1
        codebooks.append(center.tolist())
        _, index = min_distance(data_block, center)
        code.append(index.tolist())
    code = np.transpose(code)
    return np.array(codebooks,dtype = 'float32'), np.array(code,'float32')


def split_data(data, P=2,axis=1):   
    '''  
    split data into P parts  
    :param data: (N,M)
    :param P:
    :return: (P,N,M/P) A list of sub-arrays.
    '''
    return np.split(data, P, axis)

def K_means(k_centroid, data):
    '''
    K_means
    :param data: (N,M)
    :param centroid: (K,M)
    :return: new centroid (K,M)
    '''
    #print('K_means\ncentroid_shape:   ',centroid.shape)
    nearest_centroid,_ = min_distance(data, k_centroid)  #all_cluster:  N* M  N个点所属中心点的坐标
    new_centroids = np.apply_along_axis(update_centroid, 1, arr=k_centroid, all_cluster=nearest_centroid,points=data)
    return new_centroids  

def min_distance(points, min_centroids): 
    '''
    find
    :param points: (N,M)
    :param centroids: (K,M)
    :return: nearest_centroid: (N,M) each points' nearest_centroid
    '''
    distances = L1_distance(points, min_centroids)   # (N,K) N个坐标点到K个类的距离
    nearest_centroid_index = np.argmin(distances,axis=1)   # 1 * N N个点最近的中心点索引  array([0, 1, 1, 0, 1])
    nearest_centroid = min_centroids[nearest_centroid_index]  #  N* M  N个点所属中心点的坐标
    return nearest_centroid, nearest_centroid_index

def L1_distance(points,L1_centroids): 
    ''' 
    L1 distance 
    :param points: (N,M) 
    :param centroids: (K,M) 
    :return: np.array with shape(N,K) each line is the L1 distance of (point, all centroids) 
    ''' 
    return np.abs(points[:, None] - L1_centroids).sum(axis=2) 

def get_cluster_data(centroid, all_cluster, points):
    '''
    寻找属于某一个（行）centroid 的所有points 
    :param centroid:(1,M) a particular centoid 
    :param all_cluster: (N,M) 记录每个point所归属的centroid 
    :param points: all points 
    :return:（?,M）每一行代表一个point 返回所有属于这一类的点的集合 
    '''
    condition = (centroid == all_cluster) * np.ones(shape=all_cluster.shape)
    condition = (condition.sum(axis=1) == all_cluster.shape[1])
    
    correspoding_all_cluster_index = np.argwhere(condition == True)[None, :]
    correspoding_all_cluster = points[correspoding_all_cluster_index].reshape(-1,all_cluster.shape[1])

    if correspoding_all_cluster.shape[-1] > 0:
        return correspoding_all_cluster.reshape(-1, centroid.shape[0])
    else:
        return correspoding_all_cluster

def update_centroid(up_centroid,all_cluster,points):
    '''

    :param centroid:  (M)  #  (64,)
    :param all_cluster:  N* M  N个点所属中心点的坐标
    :param points: ( N,M) data
    :return: (K,M) 
    '''
    #print('update_centroid\ncentroid_shape:   ',centroid.shape)
    all_data = get_cluster_data(up_centroid,all_cluster,points)
    if len(all_data) > 0 :
        up_centroid = np.median(all_data, axis=0)
    return up_centroid


if __name__ == '__main__':
    with open(data_path, 'rb') as f:
        Data_File = pickle.load(f, encoding='bytes')
    with open(centorid_path, 'rb') as f:
        Centroids_File = pickle.load(f, encoding='bytes')
    with open(codebooks_path, 'rb') as f:
        codebooks_2 = pickle.load(f, encoding='bytes')
    with open(codes_path, 'rb') as f:
        code_2 = pickle.load(f, encoding='bytes')
    start = time.time()
    codebooks, codes = pq(data=Data_File, P=p, init_centroids=Centroids_File, max_iter=20)
    end = time.time()
    time_cost_1 = end - start
    print('p1.py')
    print(f'Runtime: {time_cost_1}')
    print('codebooks')
    print((codebooks==codebooks_2))
    print('codes')
    print((codes==code_2))




