import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

def plot_metric_distributions(rules):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(rules['support'], bins=20, color='blue')
    plt.title('Support Distribution')

    plt.subplot(1, 3, 2)
    plt.hist(rules['confidence'], bins=20, color='green')
    plt.title('Confidence Distribution')

    plt.subplot(1, 3, 3)
    plt.hist(rules['lift'], bins=20, color='red')
    plt.title('Lift Distribution')

    plt.tight_layout()
    plt.show()

def plot_top_rules(rules, metric='lift', top_n=10):
    top_rules = rules.sort_values(by=metric, ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_rules)), top_rules[metric], color='purple')
    plt.yticks(range(len(top_rules)), [f"{list(eval(ant))} -> {list(eval(con))}" for ant, con in zip(top_rules['antecedents'], top_rules['consequents'])])
    plt.xlabel(metric.capitalize())
    plt.title(f'Top {top_n} Association Rules by {metric.capitalize()}')
    plt.gca().invert_yaxis()
    plt.show()

def plot_cooccurrence_heatmap(rules):
    itemsets = []
    for ant, con in zip(rules['antecedents'], rules['consequents']):
        antecedents = eval(ant)
        consequents = eval(con)
        if isinstance(antecedents, frozenset) and isinstance(consequents, frozenset):
            itemsets.append(antecedents | consequents)

    if not itemsets:
        print("No valid itemsets found for co-occurrence matrix.")
        return

    items = sorted(set().union(*itemsets))
    if not items:
        print("No items found for co-occurrence matrix.")
        return

    co_occurrence = pd.DataFrame(0, index=items, columns=items, dtype=int)

    for itemset in itemsets:
        for item in itemset:
            for other_item in itemset:
                if item != other_item:
                    co_occurrence.at[item, other_item] += 1

    if co_occurrence.empty:
        print("Co-occurrence matrix is empty.")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(co_occurrence, cmap='viridis')
    plt.title('Item Co-occurrence Heatmap')
    plt.show()

def plot_network_graph(rules, metric='lift', top_n=10):
    top_rules = rules.sort_values(by=metric, ascending=False).head(top_n)

    G = nx.DiGraph()

    for _, rule in top_rules.iterrows():
        ant = tuple(eval(rule['antecedents']))
        con = tuple(eval(rule['consequents']))
        G.add_edge(ant, con, weight=rule[metric])

    pos = nx.spring_layout(G, k=1)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", edge_color=range(len(G.edges())), edge_cmap=plt.cm.Blues)
    plt.title('Network Graph of Top Association Rules')
    plt.show()

def analyze_rules(file_path):
    rules = pd.read_csv(file_path)

    print("First few rows of the rules:")
    print(rules.head())

    print("\nBasic statistics of the rules:")
    print(rules[['support', 'confidence', 'lift']].describe())

    plot_metric_distributions(rules)
    
    plot_top_rules(rules, metric='lift', top_n=10)
    
    plot_cooccurrence_heatmap(rules)
    
    plot_network_graph(rules, metric='lift', top_n=10)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rules_file_path = os.path.join(current_dir, '../results/apriori_rules.csv')
    
    analyze_rules(rules_file_path)
