import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

data = [['A', 'B', 'C', 'D'],
        ['B', 'C', 'E'],
        ['A', 'B', 'C', 'E'],
        ['B', 'D', 'E'],
        ['A', 'B', 'C', 'D']]

te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)
print("==================")

rules = association_rules(frequent_itemsets, min_threshold=0.6)
print(rules)
