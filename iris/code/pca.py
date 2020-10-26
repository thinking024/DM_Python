import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

data = load_iris()  # 加载数据，x表示数据集中的属性数据，y表示数据标签
x = data['data']
y = data['target']

pca = dp.PCA(n_components=0.95)  # 加载pca算法，设置降维后主成分数目为2或者贡献度》95%
reduced_x = pca.fit_transform(x)  # 对原始数据进行降维，保存在reduced_x中

for c, i, target_name in zip('rgb', [0, 1, 2], data['target_names']):
    plt.scatter(reduced_x[y == i, 0], reduced_x[y == i, 1], c=c, label=target_name)

plt.legend()
plt.show()
