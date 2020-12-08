from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import decomposition as dp
from sklearn.model_selection import train_test_split, cross_val_score  # 划分数据 交叉验证
import matplotlib.pyplot as plt
def cross_validation(data, target):
    train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=3)

    # tree.plot_tree(clf, feature_names=features, class_names=classes, filled=True)
    # plt.show()
    max_depth_range = range(2, 8)
    scores = []

    for max_depth in max_depth_range:
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        clf = clf.fit(train_X, train_y)
        # cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
        score = cross_val_score(clf, train_X, train_y, cv=10, scoring='accuracy')
        scores.append(score.mean())

    best_max_depth = 2 + scores.index(max(scores))
    print(scores, best_max_depth)
    best_clf = tree.DecisionTreeClassifier(max_depth=(best_max_depth))
    best_clf.fit(train_X, train_y)
    print("best test score:", best_clf.score(test_X, test_y))

    return max_depth_range, scores


if __name__ == '__main__':
    iris = load_iris()
    data = iris['data']
    target = iris['target']

    pca = dp.PCA(n_components=0.95)
    data_pca = pca.fit_transform(data)

    max_depth_range, scores = cross_validation(data, target)
    max_depth_range, scores_pca = cross_validation(data_pca, target)

    plt.plot(max_depth_range, scores, '-r', label='without pca')
    plt.plot(max_depth_range, scores_pca, '-g', label='after pca')
    plt.xlabel('Max_depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()