#!/usr/bin/env python3
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # === Argument check ===
    if len(sys.argv) != 3:
        print("Usage: python visualize_model_performance.py <path_to_csv> <output_folder>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_csv):
        print(f"Error: file not found: {input_csv}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # === Load and clean data ===
    df = pd.read_csv(input_csv, sep="\t|,", engine="python")
    df.columns = [c.strip() for c in df.columns]
    
    # === Ensure numeric columns ===
    run_cols = [c for c in df.columns if c.lower().startswith("run")]
    for c in run_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # === Compute averages/std/se/ci if not present ===
    if "Average Run" not in df.columns or df["Average Run"].isna().any():
        df["Average Run"] = df[run_cols].mean(axis=1)
    if "Std Run" not in df.columns or df["Std Run"].isna().any():
        df["Std Run"] = df[run_cols].std(axis=1)
    
    # Calculate standard error (SE = std / sqrt(n))
    n_runs = 4
    df["SE Run"] = df["Std Run"] / (n_runs ** 0.5)
    
    # Calculate 95% confidence interval (CI = SE * t-critical)
    # Using t-distribution with n-1 degrees of freedom
    from scipy import stats
    t_critical = stats.t.ppf(0.975, n_runs - 1)  # 95% CI, two-tailed
    df["CI_95 Run"] = df["SE Run"] * t_critical
    
    # === Plot settings ===
    sns.set(style="whitegrid", font_scale=1.0)
    model_col = "HEATMAP OR MLP"
    loss_col = "LOSS FUNCTION"
    frame_col = "NUM_FRAMES"
    avg_col = "Average Run"
    std_col = "Std Run"
    se_col = "SE Run"
    ci_col = "CI_95 Run"
    
    # Choose error type: 'std', 'se', or 'ci'
    error_type = 'ci'  # Change this to 'std' or 'se' if desired
    error_col = ci_col if error_type == 'ci' else (se_col if error_type == 'se' else std_col)
    error_label = "95% CI" if error_type == 'ci' else ("SE" if error_type == 'se' else "Std")
    
    # Get unique models and loss functions (excluding NaN values)
    models = sorted([m for m in df[model_col].unique() if pd.notna(m)])
    n_models = len(models)
    loss_functions = [l for l in df[loss_col].unique() if pd.notna(l)]
    
    print(f"Found {n_models} model types: {models}")
    print(f"Found {len(loss_functions)} loss functions: {list(loss_functions)}")
    
    # Filter out rows with NaN in model column
    df = df[df[model_col].notna()]
    
    # === Combined Plot 1: Bar plots side-by-side ===
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]  # Make it iterable if only one model
    
    for idx, model in enumerate(models):
        model_df = df[df[model_col] == model]
        
        # Prepare data for manual error bars (seaborn barplot doesn't show our custom errors)
        x_positions = []
        y_values = []
        errors = []
        colors = []
        labels_list = []
        
        # Get unique frame values and loss functions
        frames = sorted(model_df[frame_col].unique())
        losses = sorted(model_df[loss_col].unique())
        n_losses = len(losses)
        
        # Create color palette
        palette = sns.color_palette("husl", n_losses)
        color_map = {loss: palette[i] for i, loss in enumerate(losses)}
        
        # Calculate positions for grouped bars
        bar_width = 0.8 / n_losses
        for frame_idx, frame in enumerate(frames):
            for loss_idx, loss in enumerate(losses):
                subset = model_df[(model_df[frame_col] == frame) & (model_df[loss_col] == loss)]
                if len(subset) > 0:
                    x_pos = frame_idx + (loss_idx - n_losses/2 + 0.5) * bar_width
                    x_positions.append(x_pos)
                    y_values.append(subset[avg_col].values[0])
                    errors.append(subset[error_col].values[0])
                    colors.append(color_map[loss])
                    labels_list.append(loss)
        
        # Plot bars with error bars
        axes[idx].bar(x_positions, y_values, width=bar_width, color=colors, 
                     edgecolor='black', linewidth=0.5)
        axes[idx].errorbar(x_positions, y_values, yerr=errors, fmt='none', 
                          ecolor='black', capsize=3, linewidth=1.5)
        
        # Set x-axis labels and ticks
        axes[idx].set_xticks(range(len(frames)))
        axes[idx].set_xticklabels(frames)
        
        axes[idx].set_title(f"{model}", fontsize=14, fontweight='bold')
        axes[idx].set_ylabel(f"Average Best Validation Error (± {error_label})" if idx == 0 else "")
        axes[idx].set_xlabel("Number of Frames")
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[loss], edgecolor='black', label=loss) 
                          for loss in losses]
        axes[idx].legend(handles=legend_elements, title="Loss Function")
    
    fig.suptitle(f"Validation Error by Architecture, Loss Function, and Number of Frames (± {error_label})", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "combined_barplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # === Combined Plot 2: Line plots side-by-side ===
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_df = df[df[model_col] == model]
        
        for loss in model_df[loss_col].unique():
            subset = model_df[model_df[loss_col] == loss]
            axes[idx].errorbar(
                subset[frame_col],
                subset[avg_col],
                yerr=subset[error_col],
                fmt='-o',
                capsize=5,
                label=loss,
                linewidth=2,
                markersize=8
            )
        
        axes[idx].set_title(f"{model}", fontsize=14, fontweight='bold')
        axes[idx].set_xlabel("Number of Frames")
        axes[idx].set_ylabel(f"Validation Error (± {error_label})" if idx == 0 else "")
        axes[idx].legend(title="Loss Function")
        axes[idx].grid(True, alpha=0.3)
    
    fig.suptitle(f"Validation Error vs. Number of Frames (Mean ± {error_label})", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "combined_lineplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # === Combined Plot 3: Boxplots side-by-side ===
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_df = df[df[model_col] == model]
        
        melted = model_df.melt(
            id_vars=[loss_col, frame_col],
            value_vars=run_cols,
            var_name="Run",
            value_name="Validation Error"
        )
        
        sns.boxplot(
            data=melted,
            x=frame_col,
            y="Validation Error",
            hue=loss_col,
            ax=axes[idx]
        )
        
        axes[idx].set_title(f"{model}", fontsize=14, fontweight='bold')
        axes[idx].set_xlabel("Number of Frames")
        axes[idx].set_ylabel("Validation Error" if idx == 0 else "")
        axes[idx].legend(title="Loss Function")
    
    fig.suptitle("Distribution of Validation Errors by Architecture and Loss Function", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "combined_boxplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # === Bonus: Combined line plot on single axis for direct comparison ===
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers for each model
    linestyles = ['-', '--', '-.', ':']  # Different styles for loss functions
    
    for model_idx, model in enumerate(models):
        model_df = df[df[model_col] == model]
        
        for loss_idx, loss in enumerate(model_df[loss_col].unique()):
            subset = model_df[model_df[loss_col] == loss]
            ax.errorbar(
                subset[frame_col],
                subset[avg_col],
                yerr=subset[error_col],
                fmt=f'{linestyles[loss_idx % len(linestyles)]}{markers[model_idx]}',
                capsize=5,
                label=f"{model} - {loss}",
                linewidth=2,
                markersize=8
            )
    
    ax.set_title(f"Direct Comparison: All Architectures and Loss Functions (± {error_label})", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Frames", fontsize=12)
    ax.set_ylabel(f"Validation Error (± {error_label})", fontsize=12)
    ax.legend(title="Architecture - Loss", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "all_combined_lineplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    print(f"\n✅ All plots saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()