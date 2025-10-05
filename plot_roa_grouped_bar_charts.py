#!/usr/bin/env python3
"""
Create grouped bar plots separating Local Dynamics vs Final State with separate plots for each threshold
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_grouped_threshold_plots():
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
    
    # Define metrics and their display names
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Separatrix_Percent']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Separatrix Percentage (%)']
    
    # Create separate plots for each threshold
    for threshold in [0.6, 0.9]:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        # Filter data for current threshold
        df_thresh = df[df['Threshold'] == threshold].copy()
        
        # Group by method and samples
        samples = [100, 500, 1000]
        x_pos = np.arange(len(samples))
        width = 0.35
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            # Get values for each method
            ld_values = []
            fs_values = []
            
            for sample in samples:
                ld_val = df_thresh[(df_thresh['Method'] == 'Local Dynamics') & 
                                  (df_thresh['Samples'] == sample)][metric].values[0]
                fs_val = df_thresh[(df_thresh['Method'] == 'Final State') & 
                                  (df_thresh['Samples'] == sample)][metric].values[0]
                ld_values.append(ld_val)
                fs_values.append(fs_val)
            
            # Create grouped bars
            bars1 = ax.bar(x_pos - width/2, ld_values, width, 
                          label='Local Dynamics', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x_pos + width/2, fs_values, width,
                          label='Final State', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_title(f'{metric_name}\n(Threshold {threshold})', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_xlabel('Number of Samples', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(samples)
            
            # Add value labels on bars
            for bars, values in [(bars1, ld_values), (bars2, fs_values)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    if metric == 'Separatrix_Percent':
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(max(ld_values), max(fs_values)) * 0.02,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Set y-axis limits
            if metric == 'Separatrix_Percent':
                max_val = max(max(ld_values), max(fs_values))
                ax.set_ylim(0, max_val * 1.25)
            else:
                min_val = min(min(ld_values), min(fs_values))
                ax.set_ylim(min_val * 0.97, 1.02)
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()
        
        # Remove the 6th subplot and use it for overall comparison
        axes[5].remove()
        
        # Create overall comparison subplot
        ax_overall = fig.add_subplot(2, 3, 6)
        
        # Calculate average performance across all metrics (excluding separatrix)
        perf_metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        ld_avg = []
        fs_avg = []
        
        for sample in samples:
            ld_sample = df_thresh[(df_thresh['Method'] == 'Local Dynamics') & 
                                 (df_thresh['Samples'] == sample)]
            fs_sample = df_thresh[(df_thresh['Method'] == 'Final State') & 
                                 (df_thresh['Samples'] == sample)]
            
            ld_avg.append(ld_sample[perf_metrics].values[0].mean())
            fs_avg.append(fs_sample[perf_metrics].values[0].mean())
        
        bars1 = ax_overall.bar(x_pos - width/2, ld_avg, width, 
                              label='Local Dynamics', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax_overall.bar(x_pos + width/2, fs_avg, width,
                              label='Final State', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bars, values in [(bars1, ld_avg), (bars2, fs_avg)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax_overall.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax_overall.set_title(f'Average Performance\n(Threshold {threshold})', fontsize=14, fontweight='bold')
        ax_overall.set_ylabel('Average Score', fontsize=12)
        ax_overall.set_xlabel('Number of Samples', fontsize=12)
        ax_overall.set_xticks(x_pos)
        ax_overall.set_xticklabels(samples)
        ax_overall.set_ylim(min(min(ld_avg), min(fs_avg)) * 0.97, 1.02)
        ax_overall.grid(True, alpha=0.3, axis='y')
        ax_overall.legend()
        
        plt.suptitle(f'ROA Classification Performance Comparison - Threshold {threshold}', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'roa_classification_threshold_{str(threshold).replace(".", "")}_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    # Create side-by-side threshold comparison for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    key_metrics = ['Accuracy', 'F1_Score', 'Separatrix_Percent']
    key_metric_names = ['Accuracy', 'F1 Score', 'Separatrix Percentage (%)']
    
    for i, (metric, metric_name) in enumerate(zip(key_metrics, key_metric_names)):
        if i >= len(axes) - 1:  # Skip if we don't have enough subplots
            break
            
        ax = axes[i]
        
        # Data for both thresholds
        samples = [100, 500, 1000]
        x_pos = np.arange(len(samples))
        width = 0.2
        
        # Get values for all combinations
        ld_06 = []
        ld_09 = []
        fs_06 = []
        fs_09 = []
        
        for sample in samples:
            ld_06.append(df[(df['Method'] == 'Local Dynamics') & 
                           (df['Samples'] == sample) & 
                           (df['Threshold'] == 0.6)][metric].values[0])
            ld_09.append(df[(df['Method'] == 'Local Dynamics') & 
                           (df['Samples'] == sample) & 
                           (df['Threshold'] == 0.9)][metric].values[0])
            fs_06.append(df[(df['Method'] == 'Final State') & 
                           (df['Samples'] == sample) & 
                           (df['Threshold'] == 0.6)][metric].values[0])
            fs_09.append(df[(df['Method'] == 'Final State') & 
                           (df['Samples'] == sample) & 
                           (df['Threshold'] == 0.9)][metric].values[0])
        
        # Create grouped bars
        bars1 = ax.bar(x_pos - 1.5*width, ld_06, width, label='Local Dynamics (0.6)', 
                      color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x_pos - 0.5*width, ld_09, width, label='Local Dynamics (0.9)', 
                      color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x_pos + 0.5*width, fs_06, width, label='Final State (0.6)', 
                      color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars4 = ax.bar(x_pos + 1.5*width, fs_09, width, label='Final State (0.9)', 
                      color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bars, values in [(bars1, ld_06), (bars2, ld_09), (bars3, fs_06), (bars4, fs_09)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric == 'Separatrix_Percent':
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_title(f'{metric_name} - All Configurations', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(samples)
        
        if metric == 'Separatrix_Percent':
            all_values = ld_06 + ld_09 + fs_06 + fs_09
            ax.set_ylim(0, max(all_values) * 1.3)
        else:
            all_values = ld_06 + ld_09 + fs_06 + fs_09
            ax.set_ylim(min(all_values) * 0.975, max(all_values) * 1.015)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
    
    # Remove unused subplot
    axes[3].remove()
    
    plt.suptitle('ROA Classification - Method and Threshold Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('roa_classification_method_threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_grouped_threshold_plots()
    print("Grouped threshold plots saved as:")
    print("- roa_classification_threshold_06_comparison.png")
    print("- roa_classification_threshold_09_comparison.png") 
    print("- roa_classification_method_threshold_comparison.png")