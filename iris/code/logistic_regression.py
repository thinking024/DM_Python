from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition as dp
from sklearn.model_selection import train_test_split, cross_val_score  # 划分数据 交叉验证


def cross_validation(data, target):
    train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=3)

    clf = LogisticRegression()
    clf = clf.fit(train_X, train_y)
    # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
    score = cross_val_score(clf, train_X, train_y, cv=10, scoring='accuracy')
    print(score.mean())

    best_clf = LogisticRegression()
    best_clf.fit(train_X, train_y)
    print("best test score:", best_clf.score(test_X, test_y))


if __name__ == '__main__':
    iris = load_iris()
    data = iris['data']
    target = iris['target']

    # data = scale(data)

    pca = dp.PCA(n_components=0.95)
    data_pca = pca.fit_transform(data)

    clf = LogisticRegression(max_iter=500)
    clf = clf.fit(data, target)
    score = cross_val_score(clf, data, target, cv=10, scoring='accuracy')
    print(score.mean())

    clf_pca = LogisticRegression(max_iter=500)
    clf_pca = clf.fit(data_pca, target)
    score_pca = cross_val_score(clf_pca, data_pca, target, cv=10, scoring='accuracy')
    print(score_pca.mean())
