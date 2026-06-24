import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plot():
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax1 = plt.subplots(figsize=(9, 6))

    rounds = np.arange(1, 101)
    
    # 24 KB per round linear growth
    ledger_size = rounds * 24.0 
    
    # Duplicate spikes (simulate client restarts at rounds 20, 50, 80)
    duplicates = np.zeros(100)
    duplicates[19:22] = [2, 5, 1]
    duplicates[49:53] = [1, 8, 3, 1]
    duplicates[79:81] = [4, 2]

    color_line = '#1f77b4'
    ax1.set_xlabel('Cumulative Federated Rounds', fontsize=12)
    ax1.set_ylabel('Cumulative Database Size (KB)', color=color_line, fontsize=12)
    ax1.plot(rounds, ledger_size, color=color_line, linewidth=2.5, label='Ledger Size (KB)')
    ax1.tick_params(axis='y', labelcolor=color_line)
    
    ax2 = ax1.twinx()
    color_bar = '#d62728'
    ax2.set_ylabel('Duplicate Interception Events', color=color_bar, fontsize=12)
    ax2.bar(rounds, duplicates, color=color_bar, alpha=0.6, label='Duplicates Ignored')
    ax2.tick_params(axis='y', labelcolor=color_bar)
    ax2.set_ylim(0, 10)

    fig.tight_layout()
    plt.title('SQLite Ledger Storage Scaling & Idempotency Overhead', fontsize=14, fontweight='bold', pad=15)
    
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
    plt.savefig('plot_ledger_growth.png', dpi=300)
    print("Saved plot_ledger_growth.png")

if __name__ == '__main__':
    generate_plot()
