# -*- encoding: utf-8 -*-
"""
@File    : another.py
@Time    : 2020-04-22 17:24
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""

def pq(data, P, init_centroids, max_iter):
    save_l = np.array_split(data, P, axis=1)
    codebooks = init_centroids
    for itern in range(max_iter):
        init_centroids1 = codebooks
        count = 0
        for i,j in zip(save_l, init_centroids1):
            dis = np.sum(abs(i[:, None, :]-j), axis=2)
            dis_min = np.argmin(dis, axis=1)  # 应该reshape(-1,1)为了简化操作忽略
            for k in range(256):
                dis_index = np.argwhere(dis_min == k)
                if dis_index.shape[0] != 0:   # if not Bad centroid
                    part_data = i[np.ix_(dis_index.ravel())] #data_block中索引行
                    data_median = np.median(part_data, axis=0)
                    codebooks[count, k] = data_median
            count += 1
    ## 求 code
    codes = np.empty(shape=[1, save_l[0].shape[0]])
    for i,j in zip(save_l, codebooks):
        dis = np.sum(abs(i[:, None, :]-j), axis=2)
        dis_min = np.argmin(dis, axis=1).reshape(1, -1)
        codes = np.concatenate((codes, dis_min), axis=0)
    codes = codes[1:, :].T.astype('uint8')
    return codebooks.astype('float32'), codes