import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import numpy as np
import os

def run_apriori_optimization(data, min_support_values, min_confidence_values):
    results = []
    for min_support in min_support_values:
        for min_confidence in min_confidence_values:
            start_time = time.time()
            frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            end_time = time.time()
            runtime = end_time - start_time
            results.append({
                'min_support': min_support,
                'min_confidence': min_confidence,
                'runtime': runtime,
                'num_frequent_itemsets': len(frequent_itemsets),
                'num_rules': len(rules)
            })
    return pd.DataFrame(results)

def run_fpgrowth_optimization(data, min_support_values, min_confidence_values):
    results = []
    for min_support in min_support_values:
        for min_confidence in min_confidence_values:
            start_time = time.time()
            frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            end_time = time.time()
            runtime = end_time - start_time
            results.append({
                'min_support': min_support,
                'min_confidence': min_confidence,
                'runtime': runtime,
                'num_frequent_itemsets': len(frequent_itemsets),
                'num_rules': len(rules)
            })
    return pd.DataFrame(results)

def create_data_subsets(data, fractions):
    subsets = []
    for fraction in fractions:
        subset = data.sample(frac=fraction, random_state=1)
        subsets.append((fraction, subset))
    return subsets

if __name__ == "__main__":
    # Load the processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../data/Processed_OnlineRetail.csv')

    # Specify data types to avoid mixed types warning
    dtype = {col: 'bool' for col in pd.read_csv(data_path, nrows=1).columns}
    dtype['InvoiceNo'] = 'str'

    data = pd.read_csv(data_path, index_col=0, dtype=dtype, low_memory=False)

    # Define parameter values
    min_support_values = [0.01, 0.02, 0.03]
    min_confidence_values = [0.2, 0.3, 0.4]

    # Run optimization for Apriori
    print("Running optimization for Apriori...")
    apriori_results = run_apriori_optimization(data, min_support_values, min_confidence_values)
    apriori_results.to_csv(os.path.join(current_dir, '../results/apriori_optimization_results.csv'), index=False)
    print("Apriori optimization complete.")

    # Run optimization for FP-Growth
    print("Running optimization for FP-Growth...")
    fpgrowth_results = run_fpgrowth_optimization(data, min_support_values, min_confidence_values)
    fpgrowth_results.to_csv(os.path.join(current_dir, '../results/fpgrowth_optimization_results.csv'), index=False)
    print("FP-Growth optimization complete.")

    # Define data subsets
    fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
    data_subsets = create_data_subsets(data, fractions)

    # Test Apriori on different subsets
    apriori_subset_results = []
    for fraction, subset in data_subsets:
        print(f"Running Apriori on subset with fraction {fraction}...")
        subset_results = run_apriori_optimization(subset, min_support_values, min_confidence_values)
        subset_results['fraction'] = fraction
        apriori_subset_results.append(subset_results)
    apriori_subset_results = pd.concat(apriori_subset_results)
    apriori_subset_results.to_csv(os.path.join(current_dir, '../results/apriori_subset_optimization_results.csv'), index=False)
    print("Apriori subset testing complete.")

    # Test FP-Growth on different subsets
    fpgrowth_subset_results = []
    for fraction, subset in data_subsets:
        print(f"Running FP-Growth on subset with fraction {fraction}...")
        subset_results = run_fpgrowth_optimization(subset, min_support_values, min_confidence_values)
        subset_results['fraction'] = fraction
        fpgrowth_subset_results.append(subset_results)
    fpgrowth_subset_results = pd.concat(fpgrowth_subset_results)
    fpgrowth_subset_results.to_csv(os.path.join(current_dir, '../results/fpgrowth_subset_optimization_results.csv'), index=False)
    print("FP-Growth subset testing complete.")

    print("Optimization and subset testing complete. Results are saved in the results directory.")
