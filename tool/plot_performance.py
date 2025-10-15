import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np

# ------------------- Configuration -------------------
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# Set chart style
sns.set_theme(style="whitegrid")
# Try different palettes: 'Set2', 'husl', 'dark'
palette = 'husl'

# ------------------- Data Preparation -------------------
df = pd.read_csv('./result/exp_result.csv', sep="\s+")  # Use \s+ to handle multiple spaces

print("Data Preview:")
print(df.head())

# ------------------- Chart Generation -------------------
# Create a large canvas with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Qwen3-4b Model Performance Comparison', fontsize=16, fontweight='bold')

# --- Chart 1: Average Latency (avg_latency) ---
latency_plot_data = df.copy()
latency_plot_data['input_max'] = latency_plot_data['input_tokens'].astype(str) + '_' + latency_plot_data['max_new_tokens'].astype(str)

sns.barplot(data=latency_plot_data, x='exp', y='avg_latency', hue='input_max', ax=axes[0,0], palette=palette)
axes[0,0].set_title('Average Latency (seconds)')
axes[0,0].set_ylabel('Latency (s)')
axes[0,0].set_xlabel('Quantization Method')
axes[0,0].tick_params(axis='x', rotation=15)
axes[0,0].legend(title='input_tokens_max_new_tokens')

# --- Chart 2: Average Throughput (avg_throughput) ---
throughput_plot_data = df.copy()
throughput_plot_data['input_max'] = throughput_plot_data['input_tokens'].astype(str) + '_' + throughput_plot_data['max_new_tokens'].astype(str)

sns.barplot(data=throughput_plot_data, x='exp', y='avg_throughput', hue='input_max', ax=axes[0,1], palette=palette)
axes[0,1].set_title('Average Throughput (tokens/second)')
axes[0,1].set_ylabel('Throughput')
axes[0,1].set_xlabel('Quantization Method')
axes[0,1].tick_params(axis='x', rotation=15)
axes[0,1].legend(title='input_tokens_max_new_tokens')

# Format y-axis as integer
axes[0,1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))

# --- Chart 3: Peak GPU Memory (peak_gpu_mem) ---
mem_plot_data = df.copy()
mem_plot_data['input_max'] = mem_plot_data['input_tokens'].astype(str) + '_' + mem_plot_data['max_new_tokens'].astype(str)

sns.barplot(data=mem_plot_data, x='exp', y='peak_gpu_mem', hue='input_max', ax=axes[1,0], palette=palette)
axes[1,0].set_title('Peak GPU Memory Usage (GB)')
axes[1,0].set_ylabel('Memory (GB)')
axes[1,0].set_xlabel('Quantization Method')
axes[1,0].tick_params(axis='x', rotation=15)
axes[1,0].legend(title='input_tokens_max_new_tokens')

# --- Chart 4: Latency Comparison Across Quantization Methods (input_tokens=3) ---
# Filter data: only input_tokens = 3
df_input3 = df[df['input_tokens'] == 3].copy()

# Plot avg_latency vs max_new_tokens for different 'exp' methods
sns.lineplot(data=df_input3, x='max_new_tokens', y='avg_latency', hue='exp', marker='o', ax=axes[1,1], palette=palette)
axes[1,1].set_title('Latency Comparison Across Quantization Methods (input_tokens=3)')
axes[1,1].set_xlabel('Max New Tokens Generated (max_new_tokens)')
axes[1,1].set_ylabel('Average Latency (seconds)')
axes[1,1].legend(title='Quantization Method')
axes[1,1].grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent overlap
plt.tight_layout()
# Make room for the main title
plt.subplots_adjust(top=0.93)

# ------------------- Save Chart -------------------
fig.savefig('qwen3_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'qwen3_performance_comparison.png'")

# ------------------- Optional: Generate Detailed Summary Table -------------------
print("\n" + "="*50)
print("Performance Summary by Configuration")
print("="*50)
summary_df = df.pivot_table(
    index=['exp', 'input_tokens'],
    columns='max_new_tokens',
    values=['avg_latency', 'avg_throughput', 'peak_gpu_mem']
)
# Rename columns for clarity
summary_df.columns = [f'{metric}_{tokens}' for metric, tokens in summary_df.columns]
summary_df = summary_df.round(4)  # Round to 4 decimal places
print(summary_df)