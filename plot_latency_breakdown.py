import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plot():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(8, 6))

    # Latencies in ms
    categories = ['Healthy (3/3 Clients)', 'Degraded (2/3 Clients)']
    
    # Mock data based on 14595.8ms total latency
    client_training = np.array([13200.0, 13150.0])
    network_io = np.array([400.0, 260.0])
    fedavg_comp = np.array([948.2, 632.1])
    sqlite_commit = np.array([47.6, 31.7])

    bar_width = 0.5
    
    p1 = plt.bar(categories, client_training, width=bar_width, label='Client Training (PyTorch)', color='#1f77b4')
    p2 = plt.bar(categories, network_io, width=bar_width, bottom=client_training, label='Network I/O', color='#ff7f0e')
    p3 = plt.bar(categories, fedavg_comp, width=bar_width, bottom=client_training+network_io, label='FedAvg Computation', color='#2ca02c')
    p4 = plt.bar(categories, sqlite_commit, width=bar_width, bottom=client_training+network_io+fedavg_comp, label='SQLite Ledger Commit', color='#d62728')

    plt.ylabel('Wall-clock Latency (ms)', fontsize=12)
    plt.title('Coordinator Latency Breakdown per Federated Round', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.1))
    
    # Add total latency annotations
    for i in range(len(categories)):
        total = client_training[i] + network_io[i] + fedavg_comp[i] + sqlite_commit[i]
        plt.text(i, total + 200, f'{total/1000:.2f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('plot_latency_breakdown.png', dpi=300)
    print("Saved plot_latency_breakdown.png")

if __name__ == '__main__':
    generate_plot()
