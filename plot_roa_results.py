#!/usr/bin/env python3
"""
Plot ROA classification results comparison
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_performance_comparison_plots():
    # Data extracted from the table
    data = {
        'Method': ['Local Dynamics', 'Local Dynamics', 'Local Dynamics', 'Local Dynamics', 
                   'Local Dynamics', 'Local Dynamics', 'Final State', 'Final State',
                   'Final State', 'Final State', 'Final State', 'Final State'],
        'Samples': [100, 100, 500, 500, 1000, 1000, 100, 100, 500, 500, 1000, 1000],
        'Threshold': [0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9],
        'Accuracy': [0.9894605116, 0.9977846193, 0.9933839164, 0.999971733, 0.9927950011, 0.9982065197,
                     0.937276, 0.915576, 0.984198, 0.984028, 0.991751, 0.993204],
        'Precision': [0.99294412, 0.9999342459, 0.9972607462, 0.9999621369, 0.9874196685, 0.9964258478,
                      0.906392, 0.910858, 0.967125, 0.976593, 0.981244, 0.988931],
        'Recall': [0.9859270397, 0.9956347099, 0.9894857304, 0.9999813298, 0.9983090715, 1,
                   0.934285, 0.866481, 0.992886, 0.98224, 0.99774, 0.993547],
        'F1_Score': [0.9894231386, 0.9977798461, 0.9933580248, 0.9999717333, 0.9928345121, 0.9982097246,
                     0.920127, 0.888115, 0.979837, 0.979408, 0.989423, 0.991234],
        'Separatrix_Percent': [23.90, 36.00, 4.61, 42.75, 1.00, 27.93, 2.10, 10.30, 0.60, 2.60, 0.30, 1.60]
    }
    
    df = pd.DataFrame(data)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ROA Classification Performance Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different methods
    colors = {'Local Dynamics': '#2E86AB', 'Final State': '#A23B72'}
    markers = {0.6: 'o', 0.9: 's'}
    
    # Plot 1: Accuracy vs Samples
    ax = axes[0, 0]
    for method in df['Method'].unique():
        for thresh in df['Threshold'].unique():
            mask = (df['Method'] == method) & (df['Threshold'] == thresh)
            subset = df[mask]
            ax.plot(subset['Samples'], subset['Accuracy'], 
                   color=colors[method], marker=markers[thresh], 
                   label=f"{method} (thresh={thresh})", 
                   linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0.9, 1.002)
    
    # Plot 2: Precision vs Samples
    ax = axes[0, 1]
    for method in df['Method'].unique():
        for thresh in df['Threshold'].unique():
            mask = (df['Method'] == method) & (df['Threshold'] == thresh)
            subset = df[mask]
            ax.plot(subset['Samples'], subset['Precision'], 
                   color=colors[method], marker=markers[thresh], 
                   linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.002)
    
    # Plot 3: Recall vs Samples
    ax = axes[0, 2]
    for method in df['Method'].unique():
        for thresh in df['Threshold'].unique():
            mask = (df['Method'] == method) & (df['Threshold'] == thresh)
            subset = df[mask]
            ax.plot(subset['Samples'], subset['Recall'], 
                   color=colors[method], marker=markers[thresh], 
                   linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.002)
    
    # Plot 4: F1 Score vs Samples
    ax = axes[1, 0]
    for method in df['Method'].unique():
        for thresh in df['Threshold'].unique():
            mask = (df['Method'] == method) & (df['Threshold'] == thresh)
            subset = df[mask]
            ax.plot(subset['Samples'], subset['F1_Score'], 
                   color=colors[method], marker=markers[thresh], 
                   linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.002)
    
    # Plot 5: Separatrix Percentage vs Samples
    ax = axes[1, 1]
    for method in df['Method'].unique():
        for thresh in df['Threshold'].unique():
            mask = (df['Method'] == method) & (df['Threshold'] == thresh)
            subset = df[mask]
            ax.plot(subset['Samples'], subset['Separatrix_Percent'], 
                   color=colors[method], marker=markers[thresh], 
                   linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Separatrix Percentage (%)')
    ax.set_title('Separatrix Points vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 45)
    
    # Plot 6: Performance comparison bar chart
    ax = axes[1, 2]
    methods = df['Method'].unique()
    thresholds = df['Threshold'].unique()
    samples_1000 = df[df['Samples'] == 1000]
    
    x = np.arange(len(methods))
    width = 0.35
    
    for i, thresh in enumerate(thresholds):
        subset = samples_1000[samples_1000['Threshold'] == thresh]
        accuracies = [subset[subset['Method'] == method]['Accuracy'].values[0] for method in methods]
        ax.bar(x + i*width, accuracies, width, label=f'Threshold {thresh}', alpha=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy (1000 samples)')
    ax.set_title('Accuracy Comparison at 1000 Samples')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.98, 1.001)
    
    plt.tight_layout()
    plt.savefig('roa_classification_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Method'],
            int(row['Samples']),
            row['Threshold'],
            f"{row['Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1_Score']:.4f}",
            f"{row['Separatrix_Percent']:.1f}%"
        ])
    
    headers = ['Method', 'Samples', 'Threshold', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Separatrix %']
    
    # Color code rows
    row_colors = []
    for _, row in df.iterrows():
        if row['Method'] == 'Local Dynamics':
            row_colors.append('#E6F3FF')  # Light blue
        else:
            row_colors.append('#FFE6F3')  # Light pink
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', 
                     cellLoc='center', rowColours=row_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('ROA Classification Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('roa_classification_results_table.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_performance_comparison_plots()
    print("Plots saved as:")
    print("- roa_classification_performance_comparison.png")
    print("- roa_classification_results_table.png")