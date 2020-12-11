from sklearn import datasets
from sklearn import preprocessing
from sklearn import decomposition as dp
from sklearn import discriminant_analysis as da
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def cross_validation(data_dict, target):
    df = pd.DataFrame(index=['KNN', 'Logistic Regression', 'Decision Tree'],
                      columns=['datatype', 'params', 'accuracy'])
    df['accuracy'] = 0
    for datatype, data in data_dict.items():
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(data,
                                                            target,
                                                            test_size=0.2,
                                                            random_state=3)
        model_dict = {
            'KNN': (KNeighborsClassifier(), {
                'n_neighbors': list(range(1, 15)),
                'p': [1, 2]
            }),
            'Logistic Regression': (
                LogisticRegression(),
                {
                    'max_iter': [300, 500, 800],
                    'solver': ['newton-cg', 'lbfgs', 'sag'], # 损失函数优化方法
                    'multi_class': ['ovr', 'multinomial']
                }),
            'Decision Tree': (DecisionTreeClassifier(), {
                'max_depth': list(range(2, 8)),
                'criterion': ['entropy', 'gini']
            })
        }  # 名称+元组

        for model_name, (model, model_params) in model_dict.items():
            # 网格搜索，进行以准确率为指标的分层10折的交叉验证
            clf = GridSearchCV(estimator=model,
                               param_grid=model_params,
                               cv=10,
                               scoring='accuracy')
            clf.fit(X_train, y_train)  # 训练
            best_model = clf.best_estimator_  # 最佳模型的对象

            # 测试
            acc = best_model.score(X_test, y_test)

            # 找到最合适的数据
            if acc > df.loc[model_name, 'accuracy']:
                df.loc[model_name, 'accuracy'] = acc
                df.loc[model_name, 'datatype'] = datatype
                df.loc[model_name, 'params'] = str(clf.best_params_)
    return df


if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris['data']
    target = iris['target']

    # 主成分分析
    pca = dp.PCA(n_components=0.95)
    data_pca = pca.fit_transform(data)

    # 线性判别分析
    lda = da.LinearDiscriminantAnalysis(n_components=2)
    data_lda = lda.fit_transform(data, target)
    # cross_validation(data_lda, target)

    # 归一化
    minmaxScaler = preprocessing.MinMaxScaler()
    data_minmax = minmaxScaler.fit_transform(data)

    # 标准化
    standardScaler = preprocessing.StandardScaler()
    data_standard = standardScaler.fit_transform(data)

    data_dict = {
        'data': data,
        'data_pca': data_pca,
        'data_lda': data_lda,
        'data_minmax': data_minmax,
        'data_standard': data_standard
    }

    pd.set_option('max_colwidth', 200)
    df = cross_validation(data_dict, target)
    print(df)