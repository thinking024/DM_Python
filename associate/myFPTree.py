import itertools


def orderTDB(tdb, mincount):
    l_1 = {}
    for itemset in tdb:
        for item in itemset:
            if item in l_1:
                l_1[item] = l_1[item] + 1
            else:
                l_1[item] = 1

    for key in list(l_1):
        if l_1[key] < mincount:
            del l_1[key]

    l_1_order = sorted(l_1.items(), key=lambda x: x[1], reverse=True)
    l_1 = dict(l_1_order)

    tdb_order = []
    for itemset in tdb:
        keys = list(l_1)
        itemset_order = []
        for key in keys:
            if key in itemset:
                itemset_order.append(key)
        tdb_order.append(itemset_order)

    return tdb_order, l_1


class Node(object):
    def __init__(self, item, parent=None):
        self.item = item
        self.parent = parent
        self.children = []
        self.count = 1

        if parent is not None:
            parent.children.append(self)

    def __str__(self):
        return str(self.item + str(self.count))

    # 回溯不包含本身和根的路径
    def itempath_from_root(self):
        path = []
        if self.parent.item == 'root':
            return path

        node = self.parent
        while node.item != 'root':
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path


class FPTree(object):
    def __init__(self, root):
        self.root = root
        self.nodes = []

    def buildTree(self, tdb_order, item_table):
        for itemset in tdb_order:
            parent = self.root
            for item in itemset:
                flag = 0
                for child in parent.children:
                    if child.item == item:
                        flag = 1
                        child.count += 1
                        parent = child
                        break

                if flag == 0:
                    node = Node(item, parent)
                    self.nodes.append(node)
                    parent = node
                    for item_list in item_table:
                        if item_list[0] == item:
                            item_list[2].append(node)

    def printTreeByBFS(self):
        queue = [root]
        while len(queue) > 0:
            for node in queue[:]:
                print(node)
                queue.remove(node)
                for child in node.children:
                    queue.append(child)

    def printTreeByDFS(self, parent):
        print(parent)
        for child in parent.children:
            self.printTreeByDFS(child)


def get_frequent_patterns(item_table):
    mydict = {}

    for item_list in item_table:
        pattern_base = []
        count = 0
        for node in item_list[2]:
            mylist = node.itempath_from_root()

            # 取所有非空条件模式
            if len(mylist) > 0:
                # 第一次
                if len(pattern_base) == 0:
                    pattern_base = mylist
                # 条件模式做交集运算
                else:
                    temp = [val for val in pattern_base if val in mylist]
                    pattern_base = temp
                    if len(pattern_base) == 0:
                        count = 0
                        break
                count = count + node.count

        # print(node.item, pattern_base, count)
        # print()

        # 根据条件模式基整理出频繁模式
        if len(pattern_base) > 0:
            for i in range(1, len(pattern_base) + 1):
                mytuple = itertools.combinations(pattern_base, i)
                for items in mytuple:
                    frequent_itemset = items + (node.item,)
                    mydict[frequent_itemset] = count
    return mydict


if __name__ == '__main__':
    tdb = [
            ['f', 'a', 'c', 'd', 'g', 'i', 'm', 'p'],
            ['a', 'b', 'c', 'f', 'l', 'o'],
            ['b', 'f', 'h', 'j', 'm', 'p'],
            ['b', 'c', 'k', 'm', 'o', 's'],
            ['a', 'f', 'c', 'e', 'l', 'n', 'o', 'p']]
    mincount = 3
    frequent_itemsets = []

    tdb_order, l_1 = orderTDB(tdb, mincount)
    frequent_itemsets.append(l_1)
    item_table = []
    for k, v in l_1.items():
        item_list = [k, v, []]
        item_table.append(item_list)

    # build tree
    root = Node('root')
    tree = FPTree(root)
    tree.buildTree(tdb_order, item_table)

    frequent_patterns = get_frequent_patterns(item_table)
    frequent_itemsets.append(frequent_patterns)

    for i in frequent_itemsets:
        print(i)
