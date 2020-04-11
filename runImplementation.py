# -*- encoding: utf-8 -*-
"""
@File    : runImplementation.py
@Time    : 2020-04-11 15:26
@Author  : Duxy Chloe
@Software: PyCharm
"""

import submission
import pickle
import time

# How to run your implementation for Part 1
with open('./toy_example/Data_File', 'rb') as f:
    Data_File = pickle.load(f, encoding = 'bytes')
with open('./toy_example/Centroids_File', 'rb') as f:
    Centroids_File = pickle.load(f, encoding = 'bytes')
start = time.time()
codebooks, codes = submission.pq(data, P=2, init_centroids=centroids, max_iter = 20)
end = time.time()
time_cost_1 = end - start


# How to run your implementation for Part 2
with open('./toy_example/Query_File', 'rb') as f:
    Query_File = pickle.load(f, encoding = 'bytes')
queries = pickle.load(Query_File, encoding = 'bytes')
start = time.time()
candidates = submission.query(queries, codebooks, codes, T=10)
end = time.time()
time_cost_2 = end - start

# output for part 2.
print(candidates)