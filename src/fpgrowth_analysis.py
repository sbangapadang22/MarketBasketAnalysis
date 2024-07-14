import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import os

def run_fpgrowth_analysis(file_path, min_support=0.01, min_confidence=0.2):
    # Load the processed data
    basket = pd.read_csv(file_path, index_col=0)

    # Apply FP-Growth algorithm to find frequent itemsets
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules

if __name__ == "__main__":
    # Path to the processed CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '../data/Processed_OnlineRetail.csv')
    output_itemsets_path = os.path.join(current_dir, '../results/fpgrowth_frequent_itemsets.csv')
    output_rules_path = os.path.join(current_dir, '../results/fpgrowth_rules.csv')

    # Run FP-Growth analysis
    frequent_itemsets, rules = run_fpgrowth_analysis(file_path)

    # Save the results
    frequent_itemsets.to_csv(output_itemsets_path, index=False)
    rules.to_csv(output_rules_path, index=False)

    print("FP-Growth analysis complete. Results are saved in the results directory.")
