#!/usr/bin/env python3
"""
Create two focused plots analyzing separatrix percentage data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_separatrix_plots():
    # Data extracted from the table
    data = {
        'Method': ['Local Dynamics', 'Local Dynamics', 'Local Dynamics', 'Local Dynamics', 
                   'Local Dynamics', 'Local Dynamics', 'Final State', 'Final State',
                   'Final State', 'Final State', 'Final State', 'Final State'],
        'Samples': [100, 100, 500, 500, 1000, 1000, 100, 100, 500, 500, 1000, 1000],
        'Threshold': [0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9],
        'Separatrix_Percent': [23.90, 36.00, 4.61, 42.75, 1.00, 27.93, 2.10, 10.30, 0.60, 2.60, 0.30, 1.60]
    }
    
    df = pd.DataFrame(data)
    
    # Create the two plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Grouped by Method (Local Dynamics vs Final State)
    samples = [100, 500, 1000]
    x_pos = np.arange(len(samples))
    width = 0.35
    
    # Get values for threshold 0.6 and 0.9
    ld_06 = []
    ld_09 = []
    fs_06 = []
    fs_09 = []
    
    for sample in samples:
        ld_06.append(df[(df['Method'] == 'Local Dynamics') & 
                       (df['Samples'] == sample) & 
                       (df['Threshold'] == 0.6)]['Separatrix_Percent'].values[0])
        ld_09.append(df[(df['Method'] == 'Local Dynamics') & 
                       (df['Samples'] == sample) & 
                       (df['Threshold'] == 0.9)]['Separatrix_Percent'].values[0])
        fs_06.append(df[(df['Method'] == 'Final State') & 
                       (df['Samples'] == sample) & 
                       (df['Threshold'] == 0.6)]['Separatrix_Percent'].values[0])
        fs_09.append(df[(df['Method'] == 'Final State') & 
                       (df['Samples'] == sample) & 
                       (df['Threshold'] == 0.9)]['Separatrix_Percent'].values[0])
    
    # Plot 1: Method comparison with grouped thresholds
    x_pos_wide = np.arange(len(samples)) * 2  # Wider spacing for better grouping
    width = 0.4
    
    bars1 = ax1.bar(x_pos_wide - 0.6, ld_06, width, label='Local Dynamics (0.6)', 
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x_pos_wide - 0.2, ld_09, width, label='Local Dynamics (0.9)', 
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax1.bar(x_pos_wide + 0.2, fs_06, width, label='Final State (0.6)', 
                   color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax1.bar(x_pos_wide + 0.6, fs_09, width, label='Final State (0.9)', 
                   color='#d62728', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars, values in [(bars1, ld_06), (bars2, ld_09), (bars3, fs_06), (bars4, fs_09)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_title('Separatrix Percentage by Method and Threshold', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Separatrix Percentage (%)', fontsize=14)
    ax1.set_xlabel('Number of Samples', fontsize=14)
    ax1.set_xticks(x_pos_wide)
    ax1.set_xticklabels(samples, fontsize=12)
    ax1.set_ylim(0, 46)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=12, loc='upper right')
    
    # Add method group labels
    ax1.text(-0.4, 40, 'Local Dynamics', ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    ax1.text(0.4, 40, 'Final State', ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Plot 2: Threshold comparison showing the effect of threshold change
    thresholds = ['0.6', '0.9']
    x_pos_thresh = np.arange(len(thresholds))
    width = 0.15
    
    # Calculate average separatrix percentage for each method and threshold
    ld_avg_06 = np.mean(ld_06)
    ld_avg_09 = np.mean(ld_09)
    fs_avg_06 = np.mean(fs_06)
    fs_avg_09 = np.mean(fs_09)
    
    ld_avg = [ld_avg_06, ld_avg_09]
    fs_avg = [fs_avg_06, fs_avg_09]
    
    # Also show individual sample values as smaller bars
    sample_colors = ['#8dd3c7', '#bebada', '#fb8072']  # Light colors for individual samples
    
    for i, sample in enumerate(samples):
        ld_vals = [ld_06[i], ld_09[i]]
        fs_vals = [fs_06[i], fs_09[i]]
        
        ax2.bar(x_pos_thresh - 0.25 + i*0.05, ld_vals, 0.04, 
               color=sample_colors[i], alpha=0.6, label=f'{sample} samples' if i == 0 else "")
        ax2.bar(x_pos_thresh + 0.25 + i*0.05, fs_vals, 0.04, 
               color=sample_colors[i], alpha=0.6)
    
    # Main bars for averages
    bars_ld = ax2.bar(x_pos_thresh - 0.2, ld_avg, width*2, label='Local Dynamics (Avg)', 
                     color='#2E86AB', alpha=0.9, edgecolor='black', linewidth=2)
    bars_fs = ax2.bar(x_pos_thresh + 0.2, fs_avg, width*2, label='Final State (Avg)', 
                     color='#A23B72', alpha=0.9, edgecolor='black', linewidth=2)
    
    # Add value labels for averages
    for bars, values in [(bars_ld, ld_avg), (bars_fs, fs_avg)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax2.set_title('Separatrix Percentage: Threshold Effect Analysis', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Separatrix Percentage (%)', fontsize=14)
    ax2.set_xlabel('Probability Threshold', fontsize=14)
    ax2.set_xticks(x_pos_thresh)
    ax2.set_xticklabels(thresholds, fontsize=12)
    ax2.set_ylim(0, 46)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=11)
    
    # Add annotations
    ax2.annotate('Higher threshold increases\nseparatrix points for\nLocal Dynamics', 
                xy=(0.8, 35), xytext=(0.5, 25),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Separatrix Percentage Analysis - ROA Classification', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('separatrix_percentage_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a second figure with line plots showing trends
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 3: Line plot showing separatrix trends by sample size
    samples = [100, 500, 1000]
    
    ax3.plot(samples, ld_06, 'o-', color='#1f77b4', linewidth=3, markersize=8, 
            label='Local Dynamics (0.6)', alpha=0.8)
    ax3.plot(samples, ld_09, 's-', color='#ff7f0e', linewidth=3, markersize=8, 
            label='Local Dynamics (0.9)', alpha=0.8)
    ax3.plot(samples, fs_06, '^-', color='#2ca02c', linewidth=3, markersize=8, 
            label='Final State (0.6)', alpha=0.8)
    ax3.plot(samples, fs_09, 'd-', color='#d62728', linewidth=3, markersize=8, 
            label='Final State (0.9)', alpha=0.8)
    
    # Add value annotations
    for i, sample in enumerate(samples):
        ax3.text(sample, ld_06[i] + 1, f'{ld_06[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        ax3.text(sample, ld_09[i] + 1, f'{ld_09[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        ax3.text(sample, fs_06[i] + 1, f'{fs_06[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        ax3.text(sample, fs_09[i] + 1, f'{fs_09[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title('Separatrix Percentage Trends vs Sample Size', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Separatrix Percentage (%)', fontsize=14)
    ax3.set_xlabel('Number of Samples', fontsize=14)
    ax3.set_ylim(0, 46)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    ax3.set_xticks(samples)
    
    # Plot 4: Delta plot showing the effect of threshold increase
    ld_delta = [ld_09[i] - ld_06[i] for i in range(len(samples))]
    fs_delta = [fs_09[i] - fs_06[i] for i in range(len(samples))]
    
    x_pos = np.arange(len(samples))
    width = 0.35
    
    bars_ld_delta = ax4.bar(x_pos - width/2, ld_delta, width, 
                           label='Local Dynamics (0.9 - 0.6)', color='#2E86AB', alpha=0.8,
                           edgecolor='black', linewidth=1)
    bars_fs_delta = ax4.bar(x_pos + width/2, fs_delta, width, 
                           label='Final State (0.9 - 0.6)', color='#A23B72', alpha=0.8,
                           edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars, values in [(bars_ld_delta, ld_delta), (bars_fs_delta, fs_delta)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 1.5,
                    f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=11, fontweight='bold')
    
    ax4.set_title('Threshold Effect: Change in Separatrix Percentage', fontsize=16, fontweight='bold', pad=20)
    ax4.set_ylabel('Change in Separatrix Percentage (%)', fontsize=14)
    ax4.set_xlabel('Number of Samples', fontsize=14)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(samples)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=12)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Set y-limits to accommodate both positive and negative values
    max_abs = max(max(abs(v) for v in ld_delta), max(abs(v) for v in fs_delta))
    ax4.set_ylim(-max_abs*1.2, max_abs*1.2)
    
    plt.suptitle('Separatrix Percentage Trends and Threshold Effects', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('separatrix_percentage_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_separatrix_plots()
    print("Separatrix analysis plots saved as:")
    print("- separatrix_percentage_analysis.png")
    print("- separatrix_percentage_trends.png")