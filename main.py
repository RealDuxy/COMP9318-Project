# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2020-04-11 15:26
@Author  : Duxy Chloe
@Software: PyCharm
"""

# import submission
import Q1
import Q2
import pickle
import time


data_path = './toy_example/example/Data_File_2'
centorid_path = './toy_example/example/Centroids_File_2'
codebooks_path = './toy_example/example/Codebooks_2'
codes_path = './toy_example/example/Codes_2'
query_path = './toy_example/example/Query_File_2'
p = 2
def runMain():
    # How to run your implementation for Part 1
    with open(data_path, 'rb') as f:
        Data_File = pickle.load(f, encoding='bytes')
    with open(centorid_path, 'rb') as f:
        Centroids_File = pickle.load(f, encoding='bytes')
    start = time.time()
    codebooks, codes = Q1.pq(data=Data_File, P=p, init_centroids=Centroids_File, max_iter=20)
    end = time.time()
    time_cost_1 = end - start
    print(f'Q1 runtime: {time_cost_1}')
    # output for part2
    print(f'codebooks: {codebooks}')
    print(f'codes: {codes}')

    with open(codebooks_path, 'rb') as f:
        codebooks = pickle.load(f, encoding='bytes')
    with open(codes_path, 'rb') as f:
        codes = pickle.load(f, encoding='bytes')
    # How to run your implementation for Part 2
    with open(query_path, 'rb') as f:
        Query_File = pickle.load(f, encoding = 'bytes')
    # queries = pickle.load(Query_File, encoding = 'bytes')
    start = time.time()
    candidates = Q2.query(Query_File, codebooks, codes, T=10)
    end = time.time()
    time_cost_2 = end - start

    print(f'Q2 runtime: {time_cost_2}')
    # output for part 2.
    print(f'candidates: {candidates}')

# runtest
if __name__ == '__main__':
    runMain()