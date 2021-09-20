'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-30 19:51:30
LastEditors: Derry
LastEditTime: 2021-07-31 00:14:05
Description: None
'''
import numpy as np
import time


def timmer(fun):
    def deco(*args, **kwargs):
        start = time.time()
        res = fun(*args, **kwargs)
        end = time.time()
        print('{} executed in {}s'.format(fun.__name__, end - start))
        return res
    return deco


@timmer
def Iteration(M, d=0.85, times=20):  # 迭代算法
    n = M.shape[0]
    R = 1/n * np.ones((n))  # PR向量
    Ones = np.ones((n))  # 平滑项
    print("init", R)
    for i in range(times):
        R = d*np.dot(M, R) + (1-d)/n*Ones  # R = dMR +（1-d)/n*1
        print(i, R)
    return R


@timmer
def Power(M, d=0.85, times=20):  # 幂法
    n = M.shape[0]
    R = 1/n * np.ones((n))
    E = np.ones((n, n))
    A = d*M + (1-d)/n*E  # R = dM + (1-d)/n*E
    print("init", R)
    for i in range(times):
        R = np.dot(A, R)/np.max(R)  # R = AR / ||R||
        print(i, R / np.sum(R))
    return R / np.sum(R)


@timmer
def Algebra(M, d=0.85):  # 代数算法
    n = M.shape[0]
    E = np.ones((n, n))
    Ones = np.ones((n))
    R = np.dot(np.linalg.inv(E-d*M), (1-d)/n*Ones)
    return R/np.sum(R)


if __name__ == "__main__":
    d = 0.8  # 阻尼因子
    M = np.array([[0, 1/2, 0, 0], [1/3, 0, 0, 1/2],
                  [1/3, 0, 1, 1/2], [1/3, 1/2, 0, 0]])  # 转移矩阵
    print("Iteration", Iteration(M, d))
    print("Power", Power(M, d))
    print("Algebra", Algebra(M, d))
