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

data_path = './toy_example/example/Data_File_2'
centorid_path = './toy_example/example/Centroids_File_2'
codebooks_path = './toy_example/example/Codebooks_2'
codes_path = './toy_example/example/Codes_2'
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
    data_blocks = split_data(data,P)

    codebooks = []
    code = []
    for i in range(P):
        centroids = init_centroids[i]
        data_block = data_blocks[i]
        epoch = 1

        while epoch <= max_iter:
            print(f'epoch :{epoch}       max_iter: {max_iter}')
            centroids = K_means(centroids, data_block)
            epoch += 1
        # now we get new_centroid = initial_centroid = (p,k,m/p) = codebooks
        codebooks.append(centroids.tolist())

        # 如果code全对，nearest_index也是对的，说明centroids按理来说应该是被更新过得
        # 检验一下centroids[i]有没有被更新
        print((centroids == init_centroids[i]).all())
        # 看起来centroid更新了？
        _, nearest_index = min_distance(data_block, centroids)
        # nearest_index = np.transpose(nearest_index)
        code.append(nearest_index.tolist())
    code = np.transpose(code)
    return np.array(codebooks), np.array(code)


def split_data(data, P=2,axis=1):
    '''
    split data into P parts
    :param data: (N,M)
    :param P:
    :return: (P,N,M/P) A list of sub-arrays.
    '''
    return np.split(data, P, axis)

def K_means(centroid, data):
    '''
    K_means
    :param data: (N,M)
    :param centroid: (K,M)
    :return: new centroid (K,M)
    '''

    all_cluster,_ = min_distance(data, centroid)
    # all_cluster = min_distance(line_point=)
    centroid = np.apply_along_axis(update_centroid, 1, arr=centroid, all_cluster=all_cluster,points=data)

    return centroid

def min_distance(points, centroids):
    '''
    find
    :param points: (N,M)
    :param centroids: (K,M)
    :return: nearest_centroid: (N,M) each points' nearest_centroid
    '''
    distances = L1_distance(points, centroids)
    nearest_centroid_index = np.argmin(distances,axis=1)
    nearest_centroid = centroids[nearest_centroid_index]
    return nearest_centroid, nearest_centroid_index

def L1_distance(points,centroids):
    '''
    L1 distance
    :param points: (N,M)
    :param centroids: (K,M)
    :return: np.array with shape(N,K) each line is the L1 distance of (point, all centroids)
    '''
    return np.abs(points[:, None] - centroids).sum(axis=2)




# def get_cluster_data(centroid,all_cluster):
#     '''
#     find all data belong to the centroid
#     :param centroid:(1,M) a particular centoid
#     :param all_cluster: (M,1)
#     :return:
#     '''
#
#     correspoding_all_cluster = np.extract(all_cluster == centroid, all_cluster)
#     # print(correspoding_all_cluster.shape)
#     if len(correspoding_all_cluster) > 0:
#         return correspoding_all_cluster.reshape(-1, centroid.shape[0])
#     else:
#         return correspoding_all_cluster


def get_cluster_data(centroid, all_cluster, points):
    '''
    寻找属于某一个（行）centroid 的所有points
    :param centroid:(1,M) a particular centoid
    :param all_cluster: (N,M) 记录每个point所归属的centroid
    :param points: all points
    :return:（?,M）每一行代表一个point
    '''
    #     print(f"getting cluster: {centroid}")
    condition = (centroid == all_cluster) * np.ones(shape=all_cluster.shape)
    #     print(condition)
    condition = (condition.sum(axis=1) == all_cluster.shape[1])
    #     print(condition)
    #     correspoding_all_cluster = np.extract(condition[:,None],all_cluster)
    #     correspoding_all_cluster = np.where(condition==True,all_cluster,condition)
    correspoding_all_cluster_index = np.argwhere(condition == True)[None, :]
    # print("行数：",correspoding_all_cluster_index)
    # print(correspoding_all_cluster_index)
    correspoding_all_cluster = points[correspoding_all_cluster_index].reshape(-1,all_cluster.shape[1])





    #     print("there",correspoding_all_cluster)
    if correspoding_all_cluster.shape[-1] > 0:
        return correspoding_all_cluster.reshape(-1, centroid.shape[0])
    else:
        return correspoding_all_cluster

def update_centroid(centroid,all_cluster,points):
    '''

    :param centroid:
    :param all_cluster:
    :return:
    '''

    all_data = get_cluster_data(centroid,all_cluster,points)
    # print(all_data.shape)
    if len(all_data) > 0 :
        # 求all_data 在每个维度的上中位数
        centroid = np.median(all_data, axis=0)
    # print(centroid.shape,"here")
    return centroid


# runtest for Q1
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
    print(f'Runtime: {time_cost_1}')
    print((codebooks))
    # print(codebooks.shape)
    print((codes==code_2).all())
    print(codes)







