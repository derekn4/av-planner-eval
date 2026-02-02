"""
Storage Layer — database.py

Stores computed metrics in SQLite so they can be queried and compared
across episodes. Demonstrates SQL proficiency (a Waymo job requirement).

Schema:
    episodes  — one row per driving episode (metadata)
    metrics   — one row per (episode, metric) pair (long format)
    metrics_wide — view joining all metrics into one row per episode
"""

import os
import sqlite3
from typing import List, Optional

import pandas as pd

from metrics import EpisodeMetrics


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "output", "metrics.db"
)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get a connection to the SQLite database, creating it if needed.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection object.
    """
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the episodes and metrics tables if they don't exist.

    Tables:
        episodes:
            episode_id  TEXT PRIMARY KEY
            duration_s  REAL
            num_agents  INTEGER

        metrics:
            episode_id    TEXT  (foreign key → episodes)
            metric_name   TEXT
            metric_value  REAL
            metric_category TEXT  ("comfort", "safety", "efficiency")
            PRIMARY KEY (episode_id, metric_name)

    Args:
        conn: SQLite connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            duration_s REAL,
            num_agents INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            episode_id    TEXT REFERENCES episodes(episode_id),
            metric_name   TEXT,
            metric_value  REAL,
            metric_category TEXT,
            PRIMARY KEY (episode_id, metric_name)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Insert operations
# ---------------------------------------------------------------------------

# Maps each metric name to its category for storage
METRIC_CATEGORIES = {
    "max_jerk": "comfort",
    "rms_jerk": "comfort",
    "max_lateral_accel": "comfort",
    "rms_lateral_accel": "comfort",
    "braking_smoothness": "comfort",
    "speed_consistency": "comfort",
    "min_ttc": "safety",
    "min_agent_distance": "safety",
    "hard_braking_count": "safety",
    "hard_braking_rate": "safety",
    "average_speed": "efficiency",
    "progress_rate": "efficiency",
}


def insert_episode(
    conn: sqlite3.Connection,
    episode_id: str,
    duration_s: float,
    num_agents: int,
) -> None:
    """Insert or replace an episode's metadata.

    Args:
        conn: SQLite connection.
        episode_id: Unique episode identifier.
        duration_s: Episode duration in seconds.
        num_agents: Number of other agents in the episode.
    """
    conn.execute(
        "INSERT OR REPLACE INTO episodes (episode_id, duration_s, num_agents) VALUES (?, ?, ?)",
        (episode_id, duration_s, num_agents),
    )


def insert_metrics(
    conn: sqlite3.Connection,
    metrics: EpisodeMetrics,
) -> None:
    """Insert all metrics for a single episode.

    Converts the EpisodeMetrics dataclass into rows in the metrics table.

    Args:
        conn: SQLite connection.
        metrics: Computed metrics for one episode.
    """
    for name, value in metrics.to_dict().items():
        if name == "episode_id":
            continue
        category = METRIC_CATEGORIES.get(name, "unknown")
        conn.execute(
            "INSERT OR REPLACE INTO metrics (episode_id, metric_name, metric_value, metric_category) "
            "VALUES (?, ?, ?, ?)",
            (metrics.episode_id, name, float(value), category),
        )
    conn.commit()


def insert_batch(
    conn: sqlite3.Connection,
    all_metrics: List[EpisodeMetrics],
    durations: List[float],
    agent_counts: List[int],
) -> None:
    """Insert metrics for multiple episodes at once.

    Args:
        conn: SQLite connection.
        all_metrics: List of EpisodeMetrics objects.
        durations: List of durations corresponding to each episode.
        agent_counts: List of agent counts corresponding to each episode.
    """
    for metrics, duration, num_agents in zip(all_metrics, durations, agent_counts):
        insert_episode(conn, metrics.episode_id, duration, num_agents)
        # insert_metrics commits per episode; we could optimize by
        # deferring commits, but for our dataset size this is fine
        insert_metrics(conn, metrics)


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------

def get_all_metrics_wide(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get all metrics in wide format (one row per episode, one column per metric).

    This is the main query for analysis — gives you a DataFrame where each
    row is an episode and each column is a metric value.

    Args:
        conn: SQLite connection.

    Returns:
        DataFrame with columns: episode_id, duration_s, num_agents, and all metric columns.
    """
    # Load metrics in long format and pivot to wide
    metrics_df = pd.read_sql("SELECT episode_id, metric_name, metric_value FROM metrics", conn)
    if metrics_df.empty:
        return pd.DataFrame()

    wide = metrics_df.pivot(
        index="episode_id", columns="metric_name", values="metric_value"
    ).reset_index()

    # Merge with episode metadata
    episodes_df = pd.read_sql("SELECT * FROM episodes", conn)
    result = episodes_df.merge(wide, on="episode_id", how="inner")
    return result


def get_metrics_by_category(
    conn: sqlite3.Connection,
    category: str,
) -> pd.DataFrame:
    """Get all metrics of a specific category.

    Args:
        conn: SQLite connection.
        category: One of "comfort", "safety", "efficiency".

    Returns:
        DataFrame with columns: episode_id, metric_name, metric_value.
    """
    return pd.read_sql(
        "SELECT episode_id, metric_name, metric_value FROM metrics WHERE metric_category = ?",
        conn,
        params=(category,),
    )


def get_episodes_with_extreme_values(
    conn: sqlite3.Connection,
    metric_name: str,
    threshold: float,
    above: bool = True,
) -> pd.DataFrame:
    """Find episodes where a metric exceeds a threshold.

    Useful for outlier detection — e.g., "which episodes have min_ttc < 2.0?"

    Args:
        conn: SQLite connection.
        metric_name: Name of the metric to filter on.
        threshold: The threshold value.
        above: If True, find values > threshold. If False, find values < threshold.

    Returns:
        DataFrame with columns: episode_id, metric_value.
    """
    op = ">" if above else "<"
    return pd.read_sql(
        f"SELECT episode_id, metric_value FROM metrics WHERE metric_name = ? AND metric_value {op} ?",
        conn,
        params=(metric_name, threshold),
    )


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing database module...")

    conn = get_connection(":memory:")  # In-memory DB for testing
    create_tables(conn)

    # Create a fake EpisodeMetrics for testing
    fake_metrics = EpisodeMetrics(
        episode_id="test_001",
        max_jerk=5.0,
        rms_jerk=2.1,
        max_lateral_accel=1.8,
        rms_lateral_accel=0.9,
        braking_smoothness=0.5,
        speed_consistency=0.3,
        min_ttc=3.2,
        min_agent_distance=4.5,
        hard_braking_count=2,
        hard_braking_rate=1.5,
        average_speed=8.3,
        progress_rate=6.1,
    )

    insert_episode(conn, "test_001", 19.5, 42)
    insert_metrics(conn, fake_metrics)

    df = get_all_metrics_wide(conn)
    print(f"\nWide metrics table ({len(df)} rows):")
    print(df.to_string())

    conn.close()
    print("\nDatabase test complete.")
