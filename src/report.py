"""
Report Generation — report.py

Generates an HTML evaluation report summarizing driving quality metrics,
statistical comparisons, and recommendations.

Sections:
  1. Executive Summary
  2. Dataset Overview
  3. Metric Distributions (histograms + summary stats)
  4. Correlation Matrix (heatmap)
  5. Outlier Episodes
  6. Recommendations
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import ComparisonResult, DescriptiveStats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

# Human-readable names and units for display
METRIC_DISPLAY = {
    "max_jerk":            ("Max Jerk",              "m/s³"),
    "rms_jerk":            ("RMS Jerk",              "m/s³"),
    "max_lateral_accel":   ("Max Lateral Accel",     "m/s²"),
    "rms_lateral_accel":   ("RMS Lateral Accel",     "m/s²"),
    "braking_smoothness":  ("Braking Smoothness",    "m/s²"),
    "speed_consistency":   ("Speed Consistency",     "ratio"),
    "min_ttc":             ("Min Time-to-Collision", "s"),
    "min_agent_distance":  ("Min Agent Distance",    "m"),
    "hard_braking_count":  ("Hard Braking Count",    "count"),
    "hard_braking_rate":   ("Hard Braking Rate",     "events/min"),
    "average_speed":       ("Average Speed",         "m/s"),
    "progress_rate":       ("Progress Rate",         "m/s"),
}


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def plot_metric_histogram(
    values: pd.Series,
    metric_name: str,
    output_path: str,
) -> str:
    """Generate and save a histogram for a single metric.

    Args:
        values: Series of metric values across episodes.
        metric_name: Internal metric name (key in METRIC_DISPLAY).
        output_path: Directory to save the plot image.

    Returns:
        Filename of the saved plot.
    """
    display_name, unit = METRIC_DISPLAY.get(metric_name, (metric_name, ""))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values.dropna(), bins=25, edgecolor="black", alpha=0.7, color="#4C72B0")

    mean_val = values.mean()
    median_val = values.median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="orange", linestyle="-", linewidth=1.5, label=f"Median: {median_val:.2f}")

    ax.set_title(f"Distribution of {display_name}", fontsize=14)
    ax.set_xlabel(f"{display_name} ({unit})", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend()

    filename = f"hist_{metric_name}.png"
    fig.savefig(os.path.join(output_path, filename), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: str,
) -> str:
    """Generate and save a correlation heatmap.

    Args:
        corr_matrix: Square correlation DataFrame from analysis.py.
        output_path: Directory to save the plot image.

    Returns:
        Filename of the saved plot.
    """
    # Map internal names to display names for axis labels
    display_labels = [
        METRIC_DISPLAY.get(c, (c, ""))[0] for c in corr_matrix.columns
    ]
    plot_corr = corr_matrix.copy()
    plot_corr.index = display_labels
    plot_corr.columns = display_labels

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        plot_corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1,
        fmt=".2f", linewidths=0.5, ax=ax,
    )
    ax.set_title("Metric Correlation Matrix", fontsize=14)

    filename = "correlation_heatmap.png"
    fig.savefig(os.path.join(output_path, filename), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_comparison_bar_chart(
    comparisons: List[ComparisonResult],
    output_path: str,
) -> str:
    """Generate a grouped bar chart comparing two groups across metrics.

    Args:
        comparisons: List of ComparisonResult from analysis.py.
        output_path: Directory to save the plot image.

    Returns:
        Filename of the saved plot.
    """
    if not comparisons:
        return ""

    labels = [METRIC_DISPLAY.get(c.metric_name, (c.metric_name, ""))[0] for c in comparisons]
    means_a = [c.group_a_mean for c in comparisons]
    means_b = [c.group_b_mean for c in comparisons]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    bars_a = ax.bar(x - width / 2, means_a, width, label=comparisons[0].group_a_name, color="#4C72B0")
    bars_b = ax.bar(x + width / 2, means_b, width, label=comparisons[0].group_b_name, color="#DD8452")

    # Add significance stars above bars
    for i, c in enumerate(comparisons):
        max_val = max(c.group_a_mean, c.group_b_mean)
        star = ""
        if c.p_value < 0.01:
            star = "**"
        elif c.p_value < 0.05:
            star = "*"
        if star:
            ax.text(i, max_val * 1.05, star, ha="center", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title("Group Comparison", fontsize=14)
    ax.legend()

    filename = "comparison_bar_chart.png"
    fig.savefig(os.path.join(output_path, filename), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return filename


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html_report(
    desc_stats: List[DescriptiveStats],
    corr_matrix: pd.DataFrame,
    metrics_df: pd.DataFrame,
    comparisons: Optional[List[ComparisonResult]] = None,
    outlier_episodes: Optional[pd.DataFrame] = None,
    output_path: str = OUTPUT_DIR,
) -> str:
    """Generate a complete HTML evaluation report.

    Sections:
      1. Executive Summary — key findings
      2. Dataset Overview — number of episodes, duration stats
      3. Metric Distributions — histogram + stats for each metric
      4. Correlation Matrix — heatmap of metric correlations
      5. Outlier Episodes — flagged extreme values
      6. Recommendations — metric quality assessment

    Args:
        desc_stats: Descriptive statistics for all metrics.
        corr_matrix: Correlation matrix DataFrame.
        metrics_df: Wide-format metrics DataFrame (for histograms).
        comparisons: Optional group comparison results.
        outlier_episodes: Optional DataFrame of outlier episodes.
        output_path: Directory to save report and images.

    Returns:
        Path to the generated HTML file.
    """
    os.makedirs(output_path, exist_ok=True)

    # Generate chart images
    hist_files = {}
    for ds in desc_stats:
        if ds.metric_name in metrics_df.columns:
            hist_files[ds.metric_name] = plot_metric_histogram(
                metrics_df[ds.metric_name], ds.metric_name, output_path
            )

    heatmap_file = plot_correlation_heatmap(corr_matrix, output_path)

    comparison_chart_file = ""
    if comparisons:
        comparison_chart_file = plot_comparison_bar_chart(comparisons, output_path)

    # Build HTML
    n_episodes = len(metrics_df)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AV Planner Evaluation Report</title>
<style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #4C72B0; padding-bottom: 10px; }}
    h2 {{ color: #2d3436; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
    h3 {{ color: #4C72B0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; font-size: 13px; }}
    th {{ background: #4C72B0; color: white; text-align: center; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    td:first-child {{ text-align: left; font-weight: bold; }}
    .sig {{ background: #d4edda; font-weight: bold; }}
    img {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; }}
    .summary-box {{ background: white; border-left: 4px solid #4C72B0; padding: 15px; margin: 15px 0; }}
</style>
</head>
<body>
<h1>AV Planner Evaluation Report</h1>

<h2>1. Executive Summary</h2>
<div class="summary-box">
<ul>
    <li>Analyzed <strong>{n_episodes}</strong> driving episodes</li>
    <li>Computed <strong>{len(desc_stats)}</strong> metrics across comfort, safety, and efficiency</li>
"""
    # Add key findings from stats
    for ds in desc_stats:
        if ds.metric_name == "min_ttc":
            html += f'    <li>Mean min TTC: <strong>{ds.mean:.1f}s</strong> (median {ds.median:.1f}s)</li>\n'
        elif ds.metric_name == "average_speed":
            html += f'    <li>Mean speed: <strong>{ds.mean:.1f} m/s</strong> ({ds.mean * 2.237:.1f} mph)</li>\n'
        elif ds.metric_name == "hard_braking_count":
            html += f'    <li>Mean hard braking events per episode: <strong>{ds.mean:.1f}</strong></li>\n'

    html += """</ul>
</div>

<h2>2. Dataset Overview</h2>
"""
    html += f"<p>Total episodes: <strong>{n_episodes}</strong></p>\n"
    if "duration_s" in metrics_df.columns:
        html += f"<p>Mean duration: <strong>{metrics_df['duration_s'].mean():.1f}s</strong></p>\n"
    if "num_agents" in metrics_df.columns:
        html += f"<p>Mean agents per episode: <strong>{metrics_df['num_agents'].mean():.0f}</strong></p>\n"

    # Descriptive stats table
    html += "<h2>3. Metric Distributions</h2>\n"
    html += _build_stats_table_html(desc_stats)

    # Individual histograms
    for ds in desc_stats:
        if ds.metric_name in hist_files:
            display_name = METRIC_DISPLAY.get(ds.metric_name, (ds.metric_name, ""))[0]
            html += f'<h3>{display_name}</h3>\n'
            html += f'<img src="{hist_files[ds.metric_name]}" alt="{display_name}">\n'

    # Correlation matrix
    html += "<h2>4. Correlation Matrix</h2>\n"
    html += f'<img src="{heatmap_file}" alt="Correlation Heatmap">\n'

    from analysis import find_redundant_metrics
    redundant = find_redundant_metrics(corr_matrix)
    if redundant:
        html += "<p><strong>Potentially redundant metric pairs (|r| > 0.85):</strong></p><ul>\n"
        for a, b, r in redundant:
            name_a = METRIC_DISPLAY.get(a, (a, ""))[0]
            name_b = METRIC_DISPLAY.get(b, (b, ""))[0]
            html += f"<li>{name_a} &harr; {name_b}: r = {r:.2f}</li>\n"
        html += "</ul>\n"
    else:
        html += "<p>No highly redundant metric pairs found.</p>\n"

    # Comparisons
    if comparisons:
        html += "<h2>5. Group Comparisons</h2>\n"
        if comparison_chart_file:
            html += f'<img src="{comparison_chart_file}" alt="Group Comparison">\n'
        html += _build_comparison_table_html(comparisons)

    # Outliers
    if outlier_episodes is not None and not outlier_episodes.empty:
        html += "<h2>6. Outlier Episodes</h2>\n"
        html += outlier_episodes.to_html(index=False, classes="outlier-table")

    html += """
<h2>Recommendations</h2>
<div class="summary-box">
<ul>
    <li>Review episodes with min TTC &lt; 1.5s for potential safety concerns</li>
    <li>Investigate episodes with high jerk values for comfort optimization</li>
    <li>Consider removing highly correlated metrics to reduce redundancy</li>
    <li>Validate metric thresholds against human perception studies</li>
</ul>
</div>

</body>
</html>"""

    report_path = os.path.join(output_path, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to {report_path}")
    return report_path


def _build_stats_table_html(desc_stats: List[DescriptiveStats]) -> str:
    """Build an HTML table of descriptive statistics.

    Args:
        desc_stats: List of DescriptiveStats objects.

    Returns:
        HTML string for the stats table.
    """
    rows = ""
    for ds in desc_stats:
        display_name = METRIC_DISPLAY.get(ds.metric_name, (ds.metric_name, ""))[0]
        rows += f"""<tr>
            <td>{display_name}</td><td>{ds.count}</td>
            <td>{ds.mean:.3f}</td><td>{ds.std:.3f}</td><td>{ds.median:.3f}</td>
            <td>{ds.p5:.3f}</td><td>{ds.p25:.3f}</td><td>{ds.p75:.3f}</td><td>{ds.p95:.3f}</td>
            <td>{ds.skewness:.2f}</td><td>{ds.kurtosis:.2f}</td>
        </tr>\n"""
    return f"""<table>
        <tr><th>Metric</th><th>N</th><th>Mean</th><th>Std</th><th>Median</th>
        <th>P5</th><th>P25</th><th>P75</th><th>P95</th><th>Skew</th><th>Kurt</th></tr>
        {rows}
    </table>"""


def _build_comparison_table_html(comparisons: List[ComparisonResult]) -> str:
    """Build an HTML table of group comparison results.

    Args:
        comparisons: List of ComparisonResult objects.

    Returns:
        HTML string for the comparison table.
    """
    if not comparisons:
        return ""
    group_a_label = comparisons[0].group_a_name
    group_b_label = comparisons[0].group_b_name
    rows = ""
    for c in comparisons:
        display_name = METRIC_DISPLAY.get(c.metric_name, (c.metric_name, ""))[0]
        sig_class = ' class="sig"' if c.significant_at_005 else ""
        sig_text = "Yes" if c.significant_at_005 else "No"
        rows += f"""<tr{sig_class}>
            <td>{display_name}</td>
            <td>{c.group_a_mean:.3f}</td><td>{c.group_b_mean:.3f}</td>
            <td>{c.mean_difference:.3f}</td><td>{c.p_value:.4f}</td>
            <td>{c.effect_size_cohens_d:.2f}</td><td>{sig_text}</td>
            <td>{c.test_used}</td>
        </tr>\n"""
    return f"""<table>
        <tr><th>Metric</th><th>{group_a_label} Mean</th><th>{group_b_label} Mean</th>
        <th>Difference</th><th>p-value</th><th>Cohen's d</th><th>Sig (p&lt;0.05)</th><th>Test</th></tr>
        {rows}
    </table>"""


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from analysis import (
        compute_all_descriptive_stats,
        compute_correlation_matrix,
    )

    # Generate synthetic data for testing
    np.random.seed(42)
    n = 50
    test_df = pd.DataFrame({
        "episode_id": [f"ep_{i}" for i in range(n)],
        "max_jerk": np.random.normal(5.0, 1.5, n),
        "rms_jerk": np.random.normal(2.0, 0.5, n),
        "max_lateral_accel": np.random.normal(1.5, 0.4, n),
        "rms_lateral_accel": np.random.normal(0.8, 0.2, n),
        "braking_smoothness": np.random.normal(0.5, 0.15, n),
        "speed_consistency": np.random.normal(0.3, 0.1, n),
        "min_ttc": np.random.exponential(5.0, n),
        "min_agent_distance": np.random.exponential(8.0, n),
        "hard_braking_count": np.random.poisson(2, n),
        "hard_braking_rate": np.random.normal(1.5, 0.5, n),
        "average_speed": np.random.normal(8.0, 2.0, n),
        "progress_rate": np.random.normal(6.0, 1.5, n),
    })

    metric_names = [m for m in METRIC_DISPLAY.keys()]

    desc_stats = compute_all_descriptive_stats(test_df, metric_names)
    corr = compute_correlation_matrix(test_df, metric_names)

    report_path = generate_html_report(
        desc_stats=desc_stats,
        corr_matrix=corr,
        metrics_df=test_df,
    )
    print(f"Report generated: {report_path}")
