import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_heatmap(df, metric, algorithm):
    pivot_table = df.pivot_table(values=metric, index='min_support', columns='min_confidence', aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f'{algorithm} - {metric} Heatmap')
    plt.xlabel('min_confidence')
    plt.ylabel('min_support')
    plt.show()

def plot_line_plots(df, metric, algorithm):
    plt.figure(figsize=(10, 8))
    for support in df['min_support'].unique():
        subset = df[df['min_support'] == support]
        plt.plot(subset['min_confidence'], subset[metric], marker='o', label=f'support={support}')
    plt.xlabel('min_confidence')
    plt.ylabel(metric)
    plt.title(f'{algorithm} - {metric} vs. min_confidence')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the optimization results for Apriori
    apriori_results_path = os.path.join(current_dir, '../results/apriori_optimization_results.csv')
    apriori_results = pd.read_csv(apriori_results_path)
    
    # Visualize the Apriori optimization results
    for metric in ['runtime', 'num_frequent_itemsets', 'num_rules']:
        plot_heatmap(apriori_results, metric, 'Apriori')
        plot_line_plots(apriori_results, metric, 'Apriori')
    
    # Load the optimization results for FP-Growth
    fpgrowth_results_path = os.path.join(current_dir, '../results/fpgrowth_optimization_results.csv')
    fpgrowth_results = pd.read_csv(fpgrowth_results_path)
    
    # Visualize the FP-Growth optimization results
    for metric in ['runtime', 'num_frequent_itemsets', 'num_rules']:
        plot_heatmap(fpgrowth_results, metric, 'FP-Growth')
        plot_line_plots(fpgrowth_results, metric, 'FP-Growth')
    
    print("Visualization complete.")
