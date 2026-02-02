"""
AV Planner Evaluation Pipeline — main.py

Runs the full pipeline end-to-end:
  1. Extract episodes from TFRecord files
  2. Compute metrics for each episode
  3. Store metrics in SQLite
  4. Run statistical analysis
  5. Generate HTML report

Usage:
    python src/main.py data/raw/
    python src/main.py data/raw/training_20s.tfrecord-00000-of-01000
"""

import os
import sys
import time

from extract import extract_all_episodes, extract_single_file
from metrics import compute_metrics, EpisodeMetrics
from database import get_connection, create_tables, insert_episode, insert_metrics, get_all_metrics_wide
from analysis import (
    compute_all_descriptive_stats,
    compute_correlation_matrix,
    find_outlier_episodes,
)
from report import generate_html_report, METRIC_DISPLAY


def run_pipeline(path: str, output_dir: str = None) -> str:
    """Run the full evaluation pipeline.

    Args:
        path: Path to a data directory or single TFRecord file.
        output_dir: Directory for output files. Defaults to project output/.

    Returns:
        Path to the generated HTML report.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Extract episodes
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Extracting episodes from TFRecord files")
    print("=" * 60)
    t0 = time.time()

    if os.path.isdir(path):
        episodes = extract_all_episodes(path)
    else:
        episodes = extract_single_file(path)

    if not episodes:
        print("No valid episodes extracted. Exiting.")
        sys.exit(1)

    print(f"Extracted {len(episodes)} episodes in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 2: Compute metrics
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Computing metrics for each episode")
    print("=" * 60)
    t0 = time.time()

    all_metrics = []
    for i, ep in enumerate(episodes):
        m = compute_metrics(ep.episode_id, ep.ego_df, ep.agents_df, ep.duration_s)
        all_metrics.append(m)
        if (i + 1) % 25 == 0 or (i + 1) == len(episodes):
            print(f"  Computed metrics for {i + 1}/{len(episodes)} episodes")

    print(f"Metrics computed in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 3: Store in SQLite
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Storing metrics in SQLite database")
    print("=" * 60)
    t0 = time.time()

    db_path = os.path.join(output_dir, "metrics.db")
    conn = get_connection(db_path)
    create_tables(conn)

    for ep, m in zip(episodes, all_metrics):
        insert_episode(conn, ep.episode_id, ep.duration_s, ep.num_agents)
        insert_metrics(conn, m)

    # Load wide-format DataFrame for analysis
    metrics_df = get_all_metrics_wide(conn)
    conn.close()

    print(f"Stored {len(all_metrics)} episodes in {db_path}")
    print(f"Database write completed in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 4: Statistical analysis
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Running statistical analysis")
    print("=" * 60)
    t0 = time.time()

    metric_names = [name for name in METRIC_DISPLAY.keys() if name in metrics_df.columns]

    desc_stats = compute_all_descriptive_stats(metrics_df, metric_names)
    corr_matrix = compute_correlation_matrix(metrics_df, metric_names)

    # Find outliers across all metrics
    all_outliers = []
    for name in metric_names:
        outliers = find_outlier_episodes(metrics_df, name, n_std=3.0)
        if not outliers.empty:
            outliers = outliers.rename(columns={name: "value"})
            outliers["metric"] = name
            all_outliers.append(outliers)

    import pandas as pd
    outlier_df = pd.concat(all_outliers, ignore_index=True) if all_outliers else pd.DataFrame()

    # Print summary
    print(f"\n  Metrics analyzed: {len(metric_names)}")
    print(f"  Outlier episodes found: {len(outlier_df)}")
    print(f"\n  Key statistics:")
    for ds in desc_stats:
        print(f"    {ds.metric_name:25s}  mean={ds.mean:8.3f}  std={ds.std:8.3f}  median={ds.median:8.3f}")

    print(f"\nAnalysis completed in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 5: Generate report
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Generating HTML report")
    print("=" * 60)
    t0 = time.time()

    report_path = generate_html_report(
        desc_stats=desc_stats,
        corr_matrix=corr_matrix,
        metrics_df=metrics_df,
        comparisons=None,  # No group comparisons yet (single dataset)
        outlier_episodes=outlier_df if not outlier_df.empty else None,
        output_path=output_dir,
    )

    print(f"Report generated in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Episodes processed: {len(episodes)}")
    print(f"  Database:           {db_path}")
    print(f"  Report:             {report_path}")
    print(f"\nOpen the report in your browser:")
    print(f"  {os.path.abspath(report_path)}")

    return report_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <path_to_data_dir_or_single_file>")
        print()
        print("Examples:")
        print("  python src/main.py data/raw/")
        print("  python src/main.py data/raw/training_20s.tfrecord-00000-of-01000")
        sys.exit(1)

    run_pipeline(sys.argv[1])
