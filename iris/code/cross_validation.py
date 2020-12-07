from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

SPECIES_LABEL_DICT = {
    'Iris-setosa': 0,  # 山鸢尾
    'Iris-versicolor': 1,  # 变色鸢尾
    'Iris-virginica': 2  # 维吉尼亚鸢尾
}

# 使用的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def main():
    iris = datasets.load_iris()  # 加载sklearn自带的数据集鸢尾花
    data = iris.data  # 数据
    target = iris.target  # 每个数据对应的标签，0、1、2

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=1 / 10,
                                                        random_state=10)

    model_dict = {
        'kNN': (KNeighborsClassifier(), {
            'n_neighbors': [5, 15, 25],
            'p': [1, 2]
        }),
        'Logistic Regression': (LogisticRegression(), {
            'C': [1e-2, 1, 1e2]
        }),
        'SVM': (SVC(), {
            'C': [1e-2, 1, 1e2]
        })
    }  # 名称+元组

    for model_name, (model, model_params) in model_dict.items():
        # 训练模型
        clf = GridSearchCV(estimator=model, param_grid=model_params, cv=5)  # 模型、参数、折数
        clf.fit(X_train, y_train)  # 训练
        best_model = clf.best_estimator_  # 最佳模型的对象

        # 验证
        acc = best_model.score(X_test, y_test)
        print('{}模型的预测准确率：{:.2f}%'.format(model_name, acc * 100))
        print('{}模型的最优参数：{}'.format(model_name, clf.best_params_))  # 最好的模型名称和参数


if __name__ == '__main__':
    main()
