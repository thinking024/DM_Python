from sklearn import datasets
from sklearn import decomposition as dp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def fun(data, target):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=3)
    model_dict = {
        'kNN': (KNeighborsClassifier(), {
            'n_neighbors': list(range(1, 15)),
            'p': [1, 2]
        }),
        'Logistic Regression': (LogisticRegression(max_iter=500), {
            'C': [1e-2, 1, 1e2], # 正则化强度
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], # 损失函数优化方法
            'multi_class': ['ovr', 'multinomial']
        }),
        'SVM': (SVC(), {
            'C': [1e-2, 1, 1e2]
        })
    }  # 名称+元组

    for model_name, (model, model_params) in model_dict.items():
        # 训练模型
        clf = GridSearchCV(estimator=model, param_grid=model_params, cv=10)  # 模型、参数、折数
        clf.fit(X_train, y_train)  # 训练
        best_model = clf.best_estimator_  # 最佳模型的对象

        # 验证
        acc = best_model.score(X_test, y_test)
        print('{}模型的测试准确率：{:.2f}%'.format(model_name, acc * 100))
        print('{}模型的最优参数：{}'.format(model_name, clf.best_params_))  # 最好的模型名称和参数


if __name__ == '__main__':
    iris = datasets.load_iris()  # 加载sklearn自带的数据集鸢尾花
    data = iris['data']
    target = iris['target']
    pca = dp.PCA(n_components=0.95)
    data_pca = pca.fit_transform(data)
    fun(data, target)
    print('=======')
    fun(data_pca, target)

