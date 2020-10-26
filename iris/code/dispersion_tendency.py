import readFile as f
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'sepal_length': f.sepal_length,
    'sepal_width': f.sepal_width,
    'petal_length': f.petal_length,
    'petal_width': f.petal_width
})
# print(data)
data.boxplot()  # pandas绘图
# plt.boxplot(x=data.values, labels=data.columns)
plt.show()

print(data.describe())

# print(max(f.sepal_length) - min(f.sepal_length))

# sepal_length_q1 = np.percentile(f.sepal_length, 25)
# print(sepal_length_q1)
# sepal_length_q3 = np.percentile(f.sepal_length, 75)
# print(sepal_length_q3)

# sepal_length_iqr = sepal_length_q3 - sepal_length_q1
# print(sepal_length_iqr)
# print('\n=====================\n')

# print(max(f.sepal_width) - min(f.sepal_width))
# print('\n=====================\n')

# print(max(f.petal_length) - min(f.petal_length))
# print('\n=====================\n')

# print(max(f.petal_width) - min(f.petal_width))
