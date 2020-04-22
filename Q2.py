# -*- encoding: utf-8 -*-
"""
@File    : Q2.py
@Time    : 2020-04-11 03:50
@Author  : Chloe Duxy
@Software: PyCharm
"""
import helper as h
import main
import pickle
import time
import numpy as np

def split_data(data, P=2,axis=1):
    '''
    split data into P parts
    :param data: (N,M)
    :param P:
    :return: (P,N,M/P) A list of sub-arrays.
    '''
    return np.split(data,P,axis)



def L2_distance(point1,point2):
    '''
    L1 distance
    :param point1: (1,M)
    :param point2: (1,M)
    :return: L1 distance: float64
    '''
    return np.sum(np.abs(point2 - point1)) 
def find_nearest(point,codebooks, codes, T = 10): 
    '''
    PQ Query working with L1 distance
    :param point: an array with shape (1,M) and dtype='float32'
    :param codebooks: an array with shape (P, K, M/P) and dtype='float32'
    :param codes: an array with shape (N, P) and dtype=='uint8'
    :param T: the minimum number of returned candidates for each day
    :return:
         a set that contains at least T integers
    '''
    #nearest_point = {}
    N,P = codes.shape
    # step 1： 在 codebooks 里面找到 q的每部分和p个中心点的距离，返回对应的k_i 及距离值 
    distances = []
    #print(codebooks.shape)
    #print(len(point))
    #print(point[0].shape)
    for i in range(P):
        distances.append(np.apply_along_axis(L2_distance, 1, arr=codebooks[i], point2 = point[i])) # (2, 256)
    distances = np.array(distances)    
    alldis = []   #[np.zeros(N,dtype='float32')] 
    for i in range(N): 
        alldis.append(0) 
        for j in range(P): 
            alldis[i] += distances[j][codes[i][j]] 
    
    nearest_point = np.where(alldis==np.min(alldis))
    da = max(alldis)
    for i in range(len(nearest_point[0])): 
            alldis[nearest_point[0][i]] = da 
    while (True): 
        if len(nearest_point[0])  >= T: 
            break  
        temp = np.where(alldis==np.min(alldis)) 
        for i in range(len(temp[0])): 
            alldis[temp[0][i]] = da 
        nearest_point = np.concatenate((nearest_point,temp),axis = 1)  
    return set(nearest_point[0]) 
def query(queries, codebooks, codes, T): 
    ''' 
    :param queries:
    :param codebooks: 
    :param codes:
    :param T:
    :return:
    '''
    candidates = []  
    N,P = codes.shape
    n,M = queries.shape
    #print(N,P)
    for i in range(n):
        point =  split_data(queries[i], P,axis=0)    
        nearest_point = find_nearest(point,codebooks, codes, T)  
        candidates.append(nearest_point)   
    return candidates 

# runtest for Q2
if __name__ == '__main__':
    main.runMain()
