import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

x,y=load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签

pca=dp.PCA(n_components=4) #加载pca算法，设置降维后主成分数目为2
reduce = pca.fit_transform(x)
# print(pca.get_covariance())  # 获得协方差矩阵
# print(pca.explained_variance_ratio_)
print(pca.explained_variance_)  # 特征值
print(pca.components_)  # 特征向量
