from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn import decomposition as dp
from sklearn.model_selection import train_test_split, cross_val_score  # 划分数据 交叉验证


def cross_validation(data, target):
    train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=3)

    clf = GaussianNB()
    clf = clf.fit(train_X, train_y)
    # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
    score = cross_val_score(clf, train_X, train_y, cv=10, scoring='accuracy')
    print(score.mean())

    best_clf = GaussianNB()
    best_clf.fit(train_X, train_y)
    print("best test score:", best_clf.score(test_X, test_y))


if __name__ == '__main__':
    iris = load_iris()
    data = iris.data
    target = iris.target
    features = iris.feature_names
    classes = iris.target_names

    # 使用pca后似乎过拟合
    pca = dp.PCA(n_components=0.95)
    data_pca = pca.fit_transform(data)

    cross_validation(data, target)
    cross_validation(data_pca, target)
