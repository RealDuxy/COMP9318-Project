import numpy as np
import main
import pickle



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
    epoch = 1
    codebooks = []
    code = []
    for i in range(P):
        centroids = init_centroids[i]
        data_block = data_blocks[i]

        while epoch <= max_iter:
            print(f'epoch :{epoch}       max_iter: {max_iter}')
            centroids = K_means(centroids, data_block)
            epoch += 1
        # now we get new_centroid = initial_centroid = (p,k,m/p) = codebooks
        codebooks.append(centroids.tolist())
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
    new_centroids = np.apply_along_axis(update_centroid, 1, arr=centroid, all_cluster=all_cluster)
    return new_centroids

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




def get_cluster_data(centroid,all_cluster):
    '''
    find all data belong to the centroid
    :param centroid:(1,M) a particular centoid
    :param all_cluster: (M,1)
    :return:
    '''

    correspoding_all_cluster = np.extract(all_cluster == centroid, all_cluster)
    if len(correspoding_all_cluster) > 0:
        return correspoding_all_cluster.reshape(-1, centroid.shape[0])
    else:
        return correspoding_all_cluster

def update_centroid(centroid,all_cluster):
    '''

    :param centroid:
    :param all_cluster:
    :return:
    '''

    all_data = get_cluster_data(centroid,all_cluster)
    # print(all_data.shape)
    if len(all_data) > 0 :
        # 求all_data 在每个维度的上中位数
        centroid = np.median(all_data, axis=0)
    # print(centroid.shape,"here")
    return centroid

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
        if len(nearest_point[0])  >= 10: 
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
if __name__ == '__main__':
    main.runMain()
