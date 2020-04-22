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
    data_blocks = split_data(data,P)
    codebooks = []
    code = []
    for i in range(P):
        centroids = init_centroids[i]
        data_block = data_blocks[i]
        epoch = 1

        while epoch <= max_iter:
            # print(f'epoch :{epoch}       max_iter: {max_iter}')
            center = K_means(centroids, data_block)
            epoch += 1
        codebooks.append(center.tolist())
        _, index = min_distance(data_block, center)
        code.append(index.tolist())
    code = np.transpose(code)
    return np.array(codebooks,dtype = 'float32'), np.array(code,dtype = 'uint8')


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
    nearest_centroid,index = min_distance(data, k_centroid) # (N,M)
    for i in range(k_centroid.shape[0]):
        if ( index == i).any():
            k_centroid[i] = np.median(data[index == i],axis = 0)
    return k_centroid

def min_distance(points, min_centroids):
    '''
    find
    :param points: (N,M)
    :param centroids: (K,M)
    :return: nearest_centroid: (N,M) each points' nearest_centroid
    '''
    distances = L1_distance(points, min_centroids)
    nearest_centroid_index = np.argmin(distances,axis=1)
    nearest_centroid = min_centroids[nearest_centroid_index]
    return nearest_centroid, nearest_centroid_index

def L1_distance(L1points,L1_centroids):
    '''
    L1 distance
    :param points: (N,M)
    :param centroids: (K,M)
    :return: np.array with shape(N,K) each line is the L1 distance of (point, all centroids)
    '''
    return np.abs(L1points[:, None] - L1_centroids).sum(axis=2)


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
    print('Duxyp1_2.py')
    print(f'Runtime: {time_cost_1}')
    print('codebooks')
    print((codebooks==codebooks_2))
    # print(codebooks.shape)
    print('codes')
    print((codes==code_2))
    #print(codes)
    #print(code_2)
    #print(codebooks)
    print('\n')

