import readFile as f
import numpy as np
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
# print(cov_matrix.shape)

# 求协方差矩阵的特征值和特征向量

# 实现降维的过程中，有两个方法
# 一种是用特征值分解，需要矩阵是方阵，
# 另一种用奇异值分解，可以是任意矩阵，而且计算量比前者少

# egvalue是指该矩阵的特征值，输出的顺序默认为从大到小
# egvector是指该矩阵对应于特征值的特征向量，按列读取
# eigenvector,eigenvalue,v = np.linalg.svd(cov_matrix)
eigenvalue, eigenvector = np.linalg.eig(cov_matrix)
# print(eigenvalue)
# print(eigenvector)

# 选取i个贡献度>95%的特征值
# 将特征向量按对应特征值从大到小按行排列成矩阵，取前i行组成矩阵
for i in range(1, len(eigenvalue)):
    if sum(eigenvalue[:i])/sum(eigenvalue) >= 0.95:
        p = eigenvector[:, :i].T
        break

# 求降维后的数据，每一列是一个变量，每个变量的顺序不变
f = p.dot(matrix)
# print(f.T)

# 绘图

iris_setosa_x, iris_setosa_y = [], []
iris_versicolor_x, iris_versicolor_y = [], []
iris_virginica_x, iris_virginica_y = [], []

for i in range(len(f[0])):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
    if i < 50:
        iris_setosa_x.append(f[0][i])
        iris_setosa_y.append(f[1][i])
    elif 50 <= i and i < 100:
        iris_versicolor_x.append(f[0][i])
        iris_versicolor_y.append(f[1][i])
    else:
        iris_virginica_x.append(f[0][i])
        iris_virginica_y.append(f[1][i])

iris_setosa = plt.scatter(iris_setosa_x, iris_setosa_y, c='r')
iris_versicolor = plt.scatter(iris_versicolor_x, iris_versicolor_y, c='b')
iris_virginica = plt.scatter(iris_virginica_x, iris_virginica_y, c='g')
plt.legend(handles=[iris_setosa, iris_versicolor, iris_virginica],
           labels=['iris_setosa', 'iris_versicolor', 'iris_virginica'],
           loc='best')
plt.show()
