import myApriori
import pandas as pd


def associate_rules(frequent_itemsets, minconfidence):
    association = []
    for k_itemsets in frequent_itemsets:
        for key in k_itemsets:
            for antecedent in myApriori.getSublist(key):
                consequent = [var for var in list(key) if var not in antecedent]
                antecedent_count = frequent_itemsets[len(antecedent) - 1][tuple(antecedent)]
                all_count = frequent_itemsets[len(key) - 1][key]
                confidence = all_count / antecedent_count

                if confidence >= minconfidence:
                    sorted_key = sorted(key)
                    sorted_antecedent = sorted(antecedent)
                    sorted_consequent = sorted(consequent)
                    rule = [sorted_key, sorted_antecedent, sorted_consequent, all_count, antecedent_count, confidence]
                    association.append(rule)

    return association


if __name__ == '__main__':
    tdb = [
            ['A', 'B', 'C', 'D'],
            ['B', 'C', 'E'],
            ['A', 'B', 'C', 'E'],
            ['B', 'D', 'E'],
            ['A', 'B', 'C', 'D']]

    frequent = myApriori.apriori(tdb, 0.4)
    rule = associate_rules(frequent, 0.6)
    column = ['itemset', 'antecedent', 'consequent', 'support', 'antecedent_support', 'confidence']
    df_rule = pd.DataFrame(rule, index=range(len(rule)), columns=column)
    print(df_rule)
