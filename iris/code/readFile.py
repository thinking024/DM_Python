file = open('iris\\data\\iris.data', 'r+')
sepal_length = []  # 花萼长度
sepal_width = []  # 花萼宽度
petal_length = []  # 花瓣长度
petal_width = []  # 花瓣宽度
species = []

for row in file:
    # print(row)
    items = row.split(',')
    sepal_length.append(float(items[0]))
    sepal_width.append(float(items[1]))
    petal_length.append(float(items[2]))
    petal_width.append(float(items[3]))
    species.append(items[4])

file.close()
