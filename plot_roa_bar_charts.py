#!/usr/bin/env python3
"""
Create comprehensive bar plots for all ROA classification metrics
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

def create_comprehensive_bar_plots():
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
    
    # Create labels for x-axis
    def create_label(row):
        method_short = row['Method'].replace(' ', '\n')
        return f"{method_short}\n{int(row['Samples'])} samples\nThresh {row['Threshold']}"
    
    df['Label'] = df.apply(create_label, axis=1)
    
    # Set up colors
    colors_ld = ['#1f77b4', '#ff7f0e']  # Blue shades for Local Dynamics
    colors_fs = ['#2ca02c', '#d62728']  # Green/Red for Final State
    
    # Create color array
    colors = []
    for _, row in df.iterrows():
        if row['Method'] == 'Local Dynamics':
            if row['Threshold'] == 0.6:
                colors.append('#1f77b4')  # Blue
            else:
                colors.append('#ff7f0e')  # Orange
        else:
            if row['Threshold'] == 0.6:
                colors.append('#2ca02c')  # Green
            else:
                colors.append('#d62728')  # Red
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define metrics to plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Separatrix_Percent']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Separatrix Percentage (%)']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = plt.subplot(3, 2, i+1)
        
        # Create bars
        bars = ax.bar(range(len(df)), df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_title(f'{metric_name} by Configuration', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)
        
        # Set x-axis labels
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Label'], rotation=45, ha='right', fontsize=9)
        
        # Add value labels on bars
        for bar, value in zip(bars, df[metric]):
            height = bar.get_height()
            if metric == 'Separatrix_Percent':
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Set y-axis limits
        if metric == 'Separatrix_Percent':
            ax.set_ylim(0, max(df[metric]) * 1.15)
        else:
            ax.set_ylim(min(df[metric]) * 0.98, 1.01)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add method separator lines
        ax.axvline(x=5.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add text annotations for method groups
        ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Local Dynamics', ha='center', va='top', 
                fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(8.5, ax.get_ylim()[1] * 0.95, 'Final State', ha='center', va='top',
                fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Create legend subplot
    ax_legend = plt.subplot(3, 2, 6)
    ax_legend.axis('off')
    
    # Create custom legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#1f77b4', alpha=0.8, label='Local Dynamics (Threshold 0.6)'),
        Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', alpha=0.8, label='Local Dynamics (Threshold 0.9)'),
        Rectangle((0, 0), 1, 1, facecolor='#2ca02c', alpha=0.8, label='Final State (Threshold 0.6)'),
        Rectangle((0, 0), 1, 1, facecolor='#d62728', alpha=0.8, label='Final State (Threshold 0.9)')
    ]
    
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=12, title='Configuration Legend', title_fontsize=14)
    
    # Add summary statistics text
    summary_text = """
    Key Findings:
    • Local Dynamics consistently outperforms Final State
    • Higher threshold (0.9) improves accuracy but increases separatrix points
    • Best performance: Local Dynamics, 500 samples, threshold 0.9 (99.997% accuracy)
    • Most stable: Local Dynamics with 1000 samples
    """
    
    ax_legend.text(0.1, 0.3, summary_text, transform=ax_legend.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('ROA Classification Performance - Comprehensive Bar Chart Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('roa_classification_comprehensive_bar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create individual metric bar plots for better visibility
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Create bars
        bars = ax.bar(range(len(df)), df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the plot
        ax.set_title(f'{metric_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(metric_name, fontsize=14)
        
        # Set x-axis labels with better formatting
        short_labels = []
        for _, row in df.iterrows():
            method_short = 'LD' if row['Method'] == 'Local Dynamics' else 'FS'
            short_labels.append(f"{method_short}\n{int(row['Samples'])}\n{row['Threshold']}")
        
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(short_labels, fontsize=11)
        
        # Add value labels on bars
        for bar, value in zip(bars, df[metric]):
            height = bar.get_height()
            if metric == 'Separatrix_Percent':
                ax.text(bar.get_x() + bar.get_width()/2., height + max(df[metric]) * 0.01,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Set y-axis limits
        if metric == 'Separatrix_Percent':
            ax.set_ylim(0, max(df[metric]) * 1.2)
        else:
            ax.set_ylim(min(df[metric]) * 0.975, 1.015)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add method separator
        ax.axvline(x=5.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add method labels
        ax.text(2.5, ax.get_ylim()[1] * 0.92, 'Local Dynamics', ha='center', va='center', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
        ax.text(8.5, ax.get_ylim()[1] * 0.92, 'Final State', ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    # Remove the 6th subplot (empty)
    axes[5].remove()
    
    # Add overall legend
    fig.legend(handles=legend_elements, loc='lower right', fontsize=12, 
               bbox_to_anchor=(0.95, 0.05), title='Configuration', title_fontsize=14)
    
    plt.suptitle('ROA Classification Metrics - Individual Bar Charts', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig('roa_classification_individual_bar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_comprehensive_bar_plots()
    print("Bar chart plots saved as:")
    print("- roa_classification_comprehensive_bar_charts.png")
    print("- roa_classification_individual_bar_charts.png")