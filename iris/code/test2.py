import readFile as f
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

# todo: 数据标准化，平均值四舍五入
sepal_length_data = np.array(f.sepal_length) - np.mean(f.sepal_length)
sepal_width_data = np.array(f.sepal_width) - np.mean(f.sepal_width)
petal_length_data = np.array(f.petal_length) - np.mean(f.petal_length)
petal_width_data = np.array(f.petal_width) - np.mean(f.petal_width)

matrix = np.array([sepal_length_data, sepal_width_data, petal_length_data, petal_width_data])

# print(matrix.shape)

# 求协方差矩阵
cov_matrix = np.cov(matrix)  # todo:把每一行作为一个变量求协方差矩阵
print(cov_matrix)

# 求协方差矩阵的特征值和特征向量

# egvalue是指该矩阵的特征值，输出的顺序默认为从大到小；
# egvector是指该矩阵对应于特征值的特征向量，按列读取，第一列就是对应于最小特征值的特征向量
# eigenvalue, eigenvector = np.linalg.eig(cov_matrix)

# 实现降维的过程中，有两个方法
# 一种是用特征值分解，需要矩阵是方阵，
# 另一种用奇异值分解，可以是任意矩阵，而且计算量比前者少
eigenvector,eigenvalue,v = np.linalg.svd(cov_matrix)
# print(eigenvector)
# print(eigenvalue)

# 选取i个贡献度>95%的特征值
# 将特征向量按对应特征值从大到小按行排列成矩阵，取前i行组成矩阵
for i in range(1, len(eigenvalue)):
    if sum(eigenvalue[:i])/sum(eigenvalue) >= 0.95:
        p = eigenvector[:,:i].T
        # print(p)
        # p = eigenvector.T
        # print(p)
        # print('\n')
        # print(p[::-1])
        break

# print(p)
