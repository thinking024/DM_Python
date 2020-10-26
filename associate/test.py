import itertools


def apriori(tdb, minsupport):
    mincount = len(tdb) * minsupport
    frequent_itemsets = []

    l_1 = {}
    for itemset in tdb:
        for item in itemset:  # 每个itemset元组作为字典的key
            if (item, ) in l_1:
                l_1[(item, )] = l_1[(item, )] + 1
            else:
                l_1[(item, )] = 1
    for key in list(l_1):
        if l_1[key] < mincount:
            del l_1[key]
        else:
            l_1[key] = l_1[key] / len(tdb)
    frequent_itemsets.append(l_1)

    # 从 l_k-1 得到c_k，然后选出l_k
    k = 2
    l_k_1 = l_1
    while len(l_k_1) > 1:
        l_k = {}
        c_k = apriori_gen(l_k_1, k)

        for i in c_k:
            for itemset in tdb:
                if set(i).issubset(set(itemset)):
                    if i in l_k:
                        l_k[i] = l_k[i] + 1
                    else:
                        l_k[i] = 1

        # 不能在迭代遍历的同时增删元素
        for key in list(l_k):
            if l_k[key] < mincount:
                del l_k[key]
            else:
                l_k[key] = l_k[key] / len(tdb)
        frequent_itemsets.append(l_k)

        l_k_1 = l_k
        k = k + 1

    return frequent_itemsets

# todo: 优化成l_k_1每个频繁项 彼此并集 形成 c_k
def apriori_gen(l_k_1, k):
    print(l_k_1)
    print()
    c_k = set()
    for key1 in l_k_1:
        for key2 in l_k_1:
            itemset = set(key1).union(set(key2))
            if len(itemset) == k:
                c_k.add(tuple(itemset))

    # single_item_set = set()
    # for key in l_k_1:  # key是一个元组，从元组中选出单个元素
    #     for item in key:
    #         single_item_set.add(item)

    # # 单个元素组合成候选项目集
    # c_k = list(itertools.combinations(single_item_set, k))
    c_k = list(c_k)
    print(c_k)
    print('=====')

    # 剪枝操作
    for mytuple in c_k[:]:
        for mylist in getSublist(mytuple):
            if len(mylist) == k-1:  # 获取固定长度的非空真子集
                if tuple(mylist) not in l_k_1:  # 子集不在之前求出的频繁项目集中
                    print(mytuple, tuple(mylist))
                    c_k.remove(mytuple)
                    break
    
    print(c_k)
    return c_k


def getSublist(items):
    n = len(items)
    sublist = []
    for i in range(2 ** n):  # 子集个数，每循环一次一个子集
        combo = []
        for j in range(n):  # 用来判断二进制下标为j的位置数是否为1
            if(i >> j) % 2:
                combo.append(items[j])
        if len(combo) > 0 and len(combo) < len(items):  # 获取非空真子集
            sublist.append(combo)
        # sublist.append(combo)
    return sublist


if __name__ == '__main__':
    tdb = [
            ['A', 'B', 'C', 'D'],
            ['B', 'C', 'E'],
            ['A', 'B', 'C', 'E'],
            ['B', 'D', 'E'],
            ['A', 'B', 'C', 'D']]

    frequent = apriori(tdb, 0.4)
    # for i in frequent:
    #     print(i)
