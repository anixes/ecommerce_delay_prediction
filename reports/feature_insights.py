import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from delivery_delay_prediction.config import PROCESSED_DATA_DIR, FIGURES_DIR

def generate_feature_insights():
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    
    # Create output dir
    out_dir = FIGURES_DIR / "new_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Temporal Flags Impact (Black Friday & Holidays)
    temporal_cols = ['is_black_friday', 'is_holiday']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, col in enumerate(temporal_cols):
        delay_rates = df.groupby(col)['is_late'].mean()
        sns.barplot(x=delay_rates.index, y=delay_rates.values, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Delay Rate by {col}")
        axes[i].set_ylabel("Avg Delay Rate")
        axes[i].set_xlabel(col)
        # Add labels on bars
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.2f')
            
    plt.tight_layout()
    plt.savefig(out_dir / "temporal_impact.png")
    print(f"Saved temporal_impact.png to {out_dir}")
    plt.close()

    # 2. Logistical Backlog Distribution
    backlog_cols = ['seller_state_backlog', 'customer_state_backlog']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, col in enumerate(backlog_cols):
        sns.boxplot(data=df, x='is_late', y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Distribution of {col} (Log scale)")
        axes[i].set_ylabel("Log(Backlog + 1)")
        
    plt.tight_layout()
    plt.savefig(out_dir / "backlog_distribution.png")
    print(f"Saved backlog_distribution.png to {out_dir}")
    plt.close()

    # 3. Seller Recent Performance vs Ground Truth
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='seller_recent_delay_rate', hue='is_late', fill=True, common_norm=False, palette="magma")
    plt.title("Seller Recent Performance (Last 5 orders) vs Current Order Outcome")
    plt.xlabel("Recent Delay Rate")
    plt.savefig(out_dir / "seller_recent_perf_kde.png")
    print(f"Saved seller_recent_perf_kde.png to {out_dir}")
    plt.close()

    # 4. Customer Loyalty (Total Orders)
    df['customer_orders_binned'] = pd.cut(df['customer_total_orders'], bins=[0, 1.1, 2.1, 5.1, 100], labels=['1', '2', '3-5', '6+'])
    plt.figure(figsize=(10, 6))
    delay_by_loyalty = df.groupby('customer_orders_binned')['is_late'].mean()
    sns.barplot(x=delay_by_loyalty.index, y=delay_by_loyalty.values, palette="rocket")
    plt.title("Delay Rate by Customer Purchase History")
    plt.ylabel("Avg Delay Rate")
    plt.savefig(out_dir / "customer_loyalty_impact.png")
    print(f"Saved customer_loyalty_impact.png to {out_dir}")
    plt.close()

    # 5. Distance vs Backlog Interaction
    df['dist_bin'] = pd.qcut(df['distance_km'], q=3, labels=['Short', 'Medium', 'Long'])
    df['backlog_bin'] = pd.cut(df['seller_state_backlog'], bins=[-1, 2, 5, 20], labels=['Low', 'Med', 'High'])
    pivot_delay = df.pivot_table(index='dist_bin', columns='backlog_bin', values='is_late', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_delay, annot=True, cmap='YlOrRd', fmt=".3f")
    plt.title("Prob. of Delay: Distance vs Seller Backlog")
    plt.savefig(out_dir / "distance_backlog_interaction.png")
    print(f"Saved distance_backlog_interaction.png to {out_dir}")
    plt.close()

    # 6. Regional Hotspots
    plt.figure(figsize=(12, 6))
    top_states = df.groupby('seller_state')['is_late'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top_states.index, y=top_states.values, palette='flare')
    plt.title("Top 10 Seller States by Delay Probability")
    plt.ylabel("Delay Probability")
    plt.savefig(out_dir / "seller_state_hotspots.png")
    print(f"Saved seller_state_hotspots.png to {out_dir}")
    plt.close()


if __name__ == "__main__":
    print("Generating insights for new features...")
    generate_feature_insights()
    print("Done!")
