"""
Statistical Analysis — analysis.py

Compares driving quality across different conditions using proper
statistical methods. This is where you demonstrate quant fluency.

Key analyses:
  - Descriptive statistics (distributions, percentiles)
  - Two-group comparisons (t-tests, effect sizes)
  - Correlation analysis (which metrics move together?)
  - Outlier detection (which episodes are extreme?)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures for results
# ---------------------------------------------------------------------------

@dataclass
class DescriptiveStats:
    """Summary statistics for a single metric."""
    metric_name: str
    count: int
    mean: float
    std: float
    median: float
    p5: float     # 5th percentile
    p25: float    # 25th percentile
    p75: float    # 75th percentile
    p95: float    # 95th percentile
    skewness: float
    kurtosis: float


@dataclass
class ComparisonResult:
    """Result of comparing a metric between two groups."""
    metric_name: str
    group_a_name: str
    group_b_name: str
    group_a_mean: float
    group_b_mean: float
    mean_difference: float
    p_value: float
    effect_size_cohens_d: float
    significant_at_005: bool
    test_used: str  # "welch_t" or "mann_whitney"


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def compute_descriptive_stats(
    metrics_df: pd.DataFrame,
    metric_name: str,
) -> DescriptiveStats:
    """Compute summary statistics for a single metric across all episodes.

    Args:
        metrics_df: Wide-format DataFrame (one row per episode, one col per metric).
        metric_name: Which metric column to analyze.

    Returns:
        DescriptiveStats with mean, median, percentiles, skewness, kurtosis.
    """
    values = metrics_df[metric_name].dropna().values
    return DescriptiveStats(
        metric_name=metric_name,
        count=len(values),
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)),
        median=float(np.median(values)),
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        skewness=float(stats.skew(values)),
        kurtosis=float(stats.kurtosis(values)),
    )


def compute_all_descriptive_stats(
    metrics_df: pd.DataFrame,
    metric_names: List[str],
) -> List[DescriptiveStats]:
    """Compute descriptive stats for all metrics.

    Args:
        metrics_df: Wide-format DataFrame.
        metric_names: List of metric column names to analyze.

    Returns:
        List of DescriptiveStats, one per metric.
    """
    return [compute_descriptive_stats(metrics_df, name) for name in metric_names]


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------

def compare_groups(
    group_a: np.ndarray,
    group_b: np.ndarray,
    metric_name: str,
    group_a_name: str = "A",
    group_b_name: str = "B",
) -> ComparisonResult:
    """Compare a metric between two groups with full statistical output.

    Uses Welch's t-test (does not assume equal variance) if both groups
    have >= 20 samples and pass normality check. Falls back to
    Mann-Whitney U (non-parametric) otherwise.

    Also computes Cohen's d effect size:
      - |d| < 0.2 = negligible
      - |d| 0.2-0.5 = small
      - |d| 0.5-0.8 = medium
      - |d| > 0.8 = large

    Args:
        group_a: Array of metric values for group A.
        group_b: Array of metric values for group B.
        metric_name: Name of the metric being compared.
        group_a_name: Label for group A.
        group_b_name: Label for group B.

    Returns:
        ComparisonResult with p-value, effect size, and significance.
    """
    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    std_a = float(np.std(group_a, ddof=1))
    std_b = float(np.std(group_b, ddof=1))
    n_a, n_b = len(group_a), len(group_b)

    # Choose test: parametric (Welch's t) or non-parametric (Mann-Whitney)
    use_parametric = True
    if n_a < 20 or n_b < 20:
        use_parametric = False
    else:
        # Check normality — Shapiro-Wilk (p > 0.05 suggests normal)
        _, p_norm_a = stats.shapiro(group_a)
        _, p_norm_b = stats.shapiro(group_b)
        if p_norm_a < 0.05 or p_norm_b < 0.05:
            use_parametric = False

    if use_parametric:
        _, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
        test_used = "welch_t"
    else:
        _, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        test_used = "mann_whitney"

    # Cohen's d effect size
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 1e-9 else 0.0

    return ComparisonResult(
        metric_name=metric_name,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        group_a_mean=mean_a,
        group_b_mean=mean_b,
        mean_difference=mean_a - mean_b,
        p_value=float(p_value),
        effect_size_cohens_d=float(cohens_d),
        significant_at_005=p_value < 0.05,
        test_used=test_used,
    )


def compare_all_metrics(
    group_a_df: pd.DataFrame,
    group_b_df: pd.DataFrame,
    metric_names: List[str],
    group_a_name: str = "A",
    group_b_name: str = "B",
) -> List[ComparisonResult]:
    """Compare all metrics between two groups.

    Args:
        group_a_df: Wide-format metrics DataFrame for group A.
        group_b_df: Wide-format metrics DataFrame for group B.
        metric_names: List of metric column names to compare.
        group_a_name: Label for group A.
        group_b_name: Label for group B.

    Returns:
        List of ComparisonResult, one per metric.
    """
    results = []
    for name in metric_names:
        a = group_a_df[name].dropna().values
        b = group_b_df[name].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        results.append(compare_groups(a, b, name, group_a_name, group_b_name))
    return results


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

def apply_bonferroni_correction(
    results: List[ComparisonResult],
    alpha: float = 0.05,
) -> List[Tuple[ComparisonResult, float, bool]]:
    """Apply Bonferroni correction for multiple testing.

    When testing N metrics simultaneously, the chance of at least one
    false positive is ~N * alpha. Bonferroni corrects by dividing
    alpha by N.

    This is conservative — it reduces false positives but may miss
    real effects (reduced statistical power).

    Args:
        results: List of ComparisonResult from compare_all_metrics.
        alpha: Original significance level (default 0.05).

    Returns:
        List of (original_result, corrected_alpha, still_significant) tuples.
    """
    if not results:
        return []
    corrected_alpha = alpha / len(results)
    return [
        (r, corrected_alpha, r.p_value < corrected_alpha)
        for r in results
    ]


def apply_benjamini_hochberg(
    results: List[ComparisonResult],
    alpha: float = 0.05,
) -> List[Tuple[ComparisonResult, float, bool]]:
    """Apply Benjamini-Hochberg procedure for false discovery rate control.

    Less conservative than Bonferroni — controls the expected proportion
    of false positives among rejected hypotheses, rather than the
    probability of any false positive.

    Procedure:
      1. Sort p-values ascending
      2. For rank i (1-indexed), compute threshold = (i / N) * alpha
      3. Find largest i where p_value[i] <= threshold
      4. All results with rank <= i are significant

    Args:
        results: List of ComparisonResult from compare_all_metrics.
        alpha: Target false discovery rate (default 0.05).

    Returns:
        List of (original_result, adjusted_p_value, significant) tuples.
    """
    if not results:
        return []

    n = len(results)
    # Sort by p-value ascending, keeping track of original index
    indexed = sorted(enumerate(results), key=lambda x: x[1].p_value)

    # Compute BH adjusted p-values
    # adjusted_p[i] = p[i] * n / rank, capped at 1.0,
    # then enforced to be monotonically non-decreasing from the bottom
    adjusted_p = [0.0] * n
    for rank_idx, (orig_idx, r) in enumerate(indexed):
        rank = rank_idx + 1  # 1-indexed
        adjusted_p[orig_idx] = min(r.p_value * n / rank, 1.0)

    # Enforce monotonicity: walk backwards through sorted order,
    # each adjusted p must be <= the one after it
    sorted_orig_indices = [orig_idx for orig_idx, _ in indexed]
    for i in range(n - 2, -1, -1):
        idx_curr = sorted_orig_indices[i]
        idx_next = sorted_orig_indices[i + 1]
        adjusted_p[idx_curr] = min(adjusted_p[idx_curr], adjusted_p[idx_next])

    return [
        (results[i], adjusted_p[i], adjusted_p[i] < alpha)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    metrics_df: pd.DataFrame,
    metric_names: List[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute pairwise correlations between all metrics.

    Identifies redundant metrics (highly correlated) and independent
    metrics (low correlation). For metric suite design, you want
    metrics that capture different aspects of driving quality.

    Args:
        metrics_df: Wide-format DataFrame with metric columns.
        metric_names: List of metric column names to include.
        method: "pearson" (linear) or "spearman" (rank-based).

    Returns:
        Square DataFrame of correlation coefficients.
    """
    return metrics_df[metric_names].corr(method=method)


def find_redundant_metrics(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.85,
) -> List[Tuple[str, str, float]]:
    """Find pairs of metrics with high correlation (potentially redundant).

    If two metrics are highly correlated (|r| > threshold), they're
    measuring similar things. You might only need one of them.

    Args:
        corr_matrix: Correlation matrix from compute_correlation_matrix.
        threshold: Absolute correlation threshold for "redundant".

    Returns:
        List of (metric_a, metric_b, correlation) tuples.
    """
    redundant = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                redundant.append((cols[i], cols[j], float(r)))
    return redundant


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def find_outlier_episodes(
    metrics_df: pd.DataFrame,
    metric_name: str,
    n_std: float = 3.0,
) -> pd.DataFrame:
    """Find episodes where a metric is more than n_std standard deviations from the mean.

    These outliers may represent:
    - Genuinely dangerous driving scenarios
    - Data quality issues
    - Edge cases worth investigating

    Args:
        metrics_df: Wide-format DataFrame.
        metric_name: Which metric to check for outliers.
        n_std: Number of standard deviations for outlier threshold.

    Returns:
        DataFrame of outlier episodes with their metric values.
    """
    values = metrics_df[metric_name]
    mean = values.mean()
    std = values.std()
    if std < 1e-9:
        return pd.DataFrame()
    mask = (values - mean).abs() > n_std * std
    return metrics_df[mask][["episode_id", metric_name]].copy()


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 50

    # Simulate two groups of episodes (e.g., urban vs highway)
    urban_data = pd.DataFrame({
        "episode_id": [f"urban_{i}" for i in range(n)],
        "max_jerk": np.random.normal(5.0, 1.5, n),
        "rms_jerk": np.random.normal(2.0, 0.5, n),
        "min_ttc": np.random.exponential(5.0, n),
        "average_speed": np.random.normal(8.0, 2.0, n),
    })

    highway_data = pd.DataFrame({
        "episode_id": [f"highway_{i}" for i in range(n)],
        "max_jerk": np.random.normal(3.0, 1.0, n),
        "rms_jerk": np.random.normal(1.2, 0.3, n),
        "min_ttc": np.random.exponential(8.0, n),
        "average_speed": np.random.normal(25.0, 3.0, n),
    })

    metric_names = ["max_jerk", "rms_jerk", "min_ttc", "average_speed"]

    # Test descriptive stats
    all_data = pd.concat([urban_data, highway_data], ignore_index=True)
    desc_stats = compute_all_descriptive_stats(all_data, metric_names)
    print("Descriptive Statistics:")
    for ds in desc_stats:
        print(f"  {ds.metric_name}: mean={ds.mean:.2f}, std={ds.std:.2f}, "
              f"median={ds.median:.2f}, skew={ds.skewness:.2f}")

    # Test group comparisons
    print("\nUrban vs Highway Comparisons:")
    comparisons = compare_all_metrics(
        urban_data, highway_data, metric_names, "Urban", "Highway"
    )
    for comp in comparisons:
        sig = "*" if comp.significant_at_005 else ""
        print(f"  {comp.metric_name}: diff={comp.mean_difference:.2f}, "
              f"p={comp.p_value:.4f}{sig}, d={comp.effect_size_cohens_d:.2f}")

    # Test correlation
    print("\nCorrelation Matrix:")
    corr = compute_correlation_matrix(all_data, metric_names)
    print(corr.round(2).to_string())

    redundant = find_redundant_metrics(corr)
    if redundant:
        print(f"\nRedundant pairs (|r| > 0.85): {redundant}")
    else:
        print("\nNo redundant metric pairs found.")
