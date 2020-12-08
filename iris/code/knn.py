from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition as dp
from sklearn.model_selection import train_test_split, cross_val_score  # 划分数据 交叉验证
import matplotlib.pyplot as plt


def cross_validation(data, target):
    train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=3)

    k_range = range(1, 15)
    scores = []

    for k in k_range:
        # 默认采用p=2的闵可夫斯基距离，即欧氏距离
        knn = KNeighborsClassifier(k)
        # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
        score = cross_val_score(knn, train_X, train_y, cv=10, scoring='accuracy')
        scores.append(score.mean())

    best_k = 1 + scores.index(max(scores))
    print(scores, best_k)
    best_clf = KNeighborsClassifier(n_neighbors=best_k)
    best_clf.fit(train_X, train_y)
    print("best test score:", best_clf.score(test_X, test_y))

    return k_range, scores


if __name__ == '__main__':
    iris = load_iris()
    data = iris['data']
    target = iris['target']

    pca = dp.PCA(n_components=0.95)
    data_pca = pca.fit_transform(data)

    k_range, scores = cross_validation(data, target)
    k_range, scores_pca = cross_validation(data_pca, target)

    plt.plot(k_range, scores, '-r', label='without pca')
    plt.plot(k_range, scores_pca, '-g', label='after pca')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
