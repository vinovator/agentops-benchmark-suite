import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_dashboard():
    # 1. Load Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "../outputs/leaderboard.csv")
    
    if not os.path.exists(csv_path):
        print("❌ No data found. Run the benchmark first!")
        return

    df = pd.read_csv(csv_path)

    # 2. Set Professional Style
    sns.set_theme(style="whitegrid")
    # Use a color palette that looks good in Light & Dark mode
    palette = {"Agent A": "#95a5a6", "Agent B": "#3498db", "Agent C": "#e74c3c"}

    # 3. Create the Canvas (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('AgentOps Benchmark: Local (Llama 3) vs Cloud (Gemini)', fontsize=20, fontweight='bold')

    # --- Chart A: Pass Rate (The Headline) ---
    # Robust check: if 'passed' is boolean, convert to int for mean calculation if needed, 
    # but groupby mean works on booleans in pandas.
    success_rates = df.groupby("agent")["passed"].mean() * 100
    sns.barplot(x=success_rates.index, y=success_rates.values, ax=axes[0,0], palette=palette)
    axes[0,0].set_title("Success Rate by Architecture (%)", fontsize=14)
    axes[0,0].set_ylim(0, 100)
    for i, v in enumerate(success_rates.values):
        # Handle potential NaN or empty values
        if pd.notna(v):
            axes[0,0].text(i, v+2, f"{v:.1f}%", ha='center', fontweight='bold')

    # --- Chart B: Latency Distribution (The Cost) ---
    sns.boxplot(data=df, x="agent", y="duration_seconds", ax=axes[0,1], palette=palette)
    axes[0,1].set_title("Latency Distribution (Speed)", fontsize=14)
    axes[0,1].set_ylabel("Seconds")

    # --- Chart C: Quality vs Speed (The Tradeoff) ---
    # Scatter plot showing if slower agents are actually smarter
    # Robust column selection
    quality_col = "quality_score" if "quality_score" in df.columns else "quality_score_1_to_5"
    
    if quality_col in df.columns:
        sns.scatterplot(data=df, x="duration_seconds", y=quality_col, hue="agent", s=100, ax=axes[1,0], palette=palette)
        axes[1,0].set_title("Trade-off: Speed vs. Quality", fontsize=14)
        axes[1,0].set_xlabel("Time Taken (s)")
        axes[1,0].set_ylabel("Judge Score (1-5)")
    else:
        axes[1,0].text(0.5, 0.5, "Quality Score data missing", ha='center')

    # --- Chart D: Failure Analysis (The Insight) ---
    # Parse the semicolon-separated fail reasons to find top errors
    all_fails = []
    # Check if 'fail_reasons' column exists and handle NaNs
    if 'fail_reasons' in df.columns:
        for reasons in df[df['passed']==False]['fail_reasons']:
            if pd.notna(reasons):
                all_fails.extend([r.split(":")[0].strip() for r in str(reasons).split(";")])
    
    if all_fails:
        fail_counts = pd.Series(all_fails).value_counts().head(5)
        sns.barplot(y=fail_counts.index, x=fail_counts.values, ax=axes[1,1], color="#e74c3c")
        axes[1,1].set_title("Top 5 Failure Modes", fontsize=14)
    else:
        axes[1,1].text(0.5, 0.5, "No Failures Recorded! 🎉", ha='center')

    # 4. Save High-Res Image
    output_path = os.path.join(current_dir, "../outputs/benchmark_dashboard.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for title
    plt.savefig(output_path, dpi=300) # 300 DPI is Print Quality
    print(f"🖼️  Dashboard saved to: {output_path}")
    # plt.show() # Commented out to prevent blocking in headless environments

if __name__ == "__main__":
    generate_dashboard()
