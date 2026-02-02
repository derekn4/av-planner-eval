"""
Metric Engine — metrics.py

Computes driving quality metrics per episode across three categories:
  - Comfort:    How smooth and pleasant is the ride?
  - Safety:     How safely does the vehicle behave around other agents?
  - Efficiency: How effectively does the vehicle make progress?

Each metric returns a single scalar per episode so they can be
compared and aggregated across episodes.
"""

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class EpisodeMetrics:
    """All computed metrics for a single driving episode."""
    episode_id: str

    # --- Comfort metrics ---
    max_jerk: float                # m/s³  — worst sudden acceleration change
    rms_jerk: float                # m/s³  — overall smoothness
    max_lateral_accel: float       # m/s²  — sharpest turn force felt by rider
    rms_lateral_accel: float       # m/s²  — overall turning comfort
    braking_smoothness: float      # m/s²  — std of deceleration values
    speed_consistency: float       # ratio — coefficient of variation of speed

    # --- Safety metrics ---
    min_ttc: float                 # seconds — closest time-to-collision
    min_agent_distance: float      # meters  — closest physical proximity
    hard_braking_count: int        # count   — number of emergency braking events
    hard_braking_rate: float       # events/min — frequency of hard braking

    # --- Efficiency metrics ---
    average_speed: float           # m/s — mean speed over episode
    progress_rate: float           # m/s — straight-line displacement / time

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Comfort metrics
# ---------------------------------------------------------------------------

def _max_jerk(ego_df: pd.DataFrame) -> float:
    """Maximum absolute jerk across the episode.

    Jerk = rate of change of acceleration. High jerk means sudden
    lurches that passengers feel. Lower is more comfortable.

    Args:
        ego_df: Ego trajectory DataFrame with 'jerk' column.

    Returns:
        Max absolute jerk in m/s³.
    """
    # TODO: Return the max of the absolute value of the jerk column
    return ego_df['jerk'].abs().max()

def _rms_jerk(ego_df: pd.DataFrame) -> float:
    """Root-mean-square jerk across the episode.

    RMS is less sensitive to single-frame noise spikes than max.
    Captures overall smoothness of the ride.

    Args:
        ego_df: Ego trajectory DataFrame with 'jerk' column.

    Returns:
        RMS jerk in m/s³.
    """
    # TODO: Compute sqrt(mean(jerk²))
    return np.sqrt((ego_df['jerk'] ** 2).mean())


def _max_lateral_accel(ego_df: pd.DataFrame) -> float:
    """Maximum absolute lateral acceleration.

    This is the sideways force passengers feel in turns.
    Typical comfortable threshold: < 2.0 m/s².

    Args:
        ego_df: Ego trajectory DataFrame with 'lateral_acceleration' column.

    Returns:
        Max absolute lateral acceleration in m/s².
    """
    # TODO: Return the max of the absolute value of lateral_acceleration
    return ego_df['lateral_acceleration'].abs().max()


def _rms_lateral_accel(ego_df: pd.DataFrame) -> float:
    """Root-mean-square lateral acceleration.

    Args:
        ego_df: Ego trajectory DataFrame with 'lateral_acceleration' column.

    Returns:
        RMS lateral acceleration in m/s².
    """
    # TODO: Compute sqrt(mean(lateral_acceleration²))
    return np.sqrt((ego_df['lateral_acceleration'] ** 2).mean())


def _braking_smoothness(ego_df: pd.DataFrame) -> float:
    """Standard deviation of braking (negative acceleration) values.

    Lower std means more consistent, predictable braking.
    If no braking occurs, returns 0.

    Args:
        ego_df: Ego trajectory DataFrame with 'acceleration' column.

    Returns:
        Std of negative acceleration values in m/s².
    """
    # Filter to rows where acceleration < 0 (braking)
    # Return the std of those values, or 0 if no braking occurred
    braking = ego_df[ego_df['acceleration'] < 0]['acceleration']
    braking_std = braking.std() if len(braking) > 1 else 0.0
    return braking_std


def _speed_consistency(ego_df: pd.DataFrame) -> float:
    """Coefficient of variation of speed (std / mean).

    Lower means more consistent speed. A value of 0 means
    perfectly constant speed. Returns 0 if mean speed is ~0
    (vehicle is stationary).

    Args:
        ego_df: Ego trajectory DataFrame with 'velocity' column.

    Returns:
        Dimensionless ratio (std / mean).
    """
    # Compute std(velocity) / mean(velocity)
    # Guard against division by zero if mean speed is near 0
    mean_speed = ego_df['velocity'].mean()
    if mean_speed < 0.01:
        return 0.0
    return ego_df['velocity'].std() / mean_speed


# ---------------------------------------------------------------------------
# Safety metrics
# ---------------------------------------------------------------------------

def _min_ttc(ego_df: pd.DataFrame, agents_df: pd.DataFrame) -> float:
    """Minimum time-to-collision across all agents and timesteps.

    TTC = distance / closing_speed, only defined when closing_speed > 0
    (i.e., ego and agent are approaching each other).

    This is the most important safety metric in AV evaluation.
    Lower TTC = more dangerous. Typical thresholds:
      - TTC < 1.5s = critical
      - TTC < 3.0s = concerning
      - TTC > 5.0s = comfortable

    Args:
        ego_df: Ego trajectory with columns [timestamp, x, y, velocity_x, velocity_y].
        agents_df: All agents with columns [timestamp, x, y, velocity, agent_id].

    Returns:
        Minimum TTC in seconds. Returns float('inf') if no closing
        situations occurred (ego never approached any agent).
    """
    if agents_df.empty:
        return float("inf")

    # Merge ego and agent data on timestamp so we can vectorize.
    # Suffix ego columns with _ego to avoid collision after merge.
    merged = agents_df.merge(
        ego_df[["timestamp", "x", "y", "velocity_x", "velocity_y"]],
        on="timestamp",
        suffixes=("_agent", "_ego"),
    )
    if merged.empty:
        return float("inf")

    # Distance between ego and each agent at each timestamp
    dx = merged["x_agent"] - merged["x_ego"]
    dy = merged["y_agent"] - merged["y_ego"]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Unit vector from ego toward agent
    dist_safe = dist.clip(lower=1e-6)  # avoid division by zero
    ux = dx / dist_safe
    uy = dy / dist_safe

    # Relative velocity projected onto the ego→agent direction.
    # Positive closing speed means they are getting closer.
    # Agent velocity components: agents_df has velocity_x, velocity_y
    # but they may not always be present. Fall back to 0 if missing.
    agent_vx = merged["velocity_x_agent"] if "velocity_x_agent" in merged.columns else 0
    agent_vy = merged["velocity_y_agent"] if "velocity_y_agent" in merged.columns else 0
    ego_vx = merged["velocity_x_ego"]
    ego_vy = merged["velocity_y_ego"]

    # Closing speed = component of (ego_vel - agent_vel) along ego→agent direction
    closing_speed = (ego_vx - agent_vx) * ux + (ego_vy - agent_vy) * uy

    # TTC only defined when closing_speed > 0 (approaching)
    approaching = closing_speed > 0.1  # small threshold to filter noise
    if not approaching.any():
        return float("inf")

    ttc = dist[approaching] / closing_speed[approaching]
    return float(ttc.min())


def _min_agent_distance(ego_df: pd.DataFrame, agents_df: pd.DataFrame) -> float:
    """Minimum Euclidean distance between ego and any agent.

    A simpler safety proxy than TTC. Doesn't account for relative
    velocity — two vehicles could be close but moving apart (safe).

    Args:
        ego_df: Ego trajectory with columns [timestamp, x, y].
        agents_df: All agents with columns [timestamp, x, y, agent_id].

    Returns:
        Minimum distance in meters. Returns float('inf') if no agents.
    """
    if agents_df.empty:
        return float("inf")

    # Merge ego position with all agent positions on timestamp
    merged = agents_df.merge(
        ego_df[["timestamp", "x", "y"]],
        on="timestamp",
        suffixes=("_agent", "_ego"),
    )
    if merged.empty:
        return float("inf")

    dx = merged["x_agent"] - merged["x_ego"]
    dy = merged["y_agent"] - merged["y_ego"]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return float(dist.min())


def _hard_braking_count(ego_df: pd.DataFrame) -> int:
    """Count of hard braking events (acceleration < -3.0 m/s²).

    -3.0 m/s² is a common industry threshold for "hard braking."
    Frequent hard braking suggests the planner is reacting late
    to situations instead of planning ahead.

    Args:
        ego_df: Ego trajectory DataFrame with 'acceleration' column.

    Returns:
        Number of timesteps where acceleration < -3.0 m/s².
    """
    # Count rows where acceleration < -3.0
    return (ego_df['acceleration'] < -3.0).sum()


def _hard_braking_rate(ego_df: pd.DataFrame, duration_s: float) -> float:
    """Hard braking events per minute.

    Normalizes hard_braking_count by episode duration so episodes
    of different lengths are comparable.

    Args:
        ego_df: Ego trajectory DataFrame with 'acceleration' column.
        duration_s: Episode duration in seconds.

    Returns:
        Hard braking events per minute.
    """
    if duration_s < 0.01:
        return 0.0
    count = _hard_braking_count(ego_df)
    return count / (duration_s / 60.0)


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def _average_speed(ego_df: pd.DataFrame) -> float:
    """Mean speed over the episode.

    Args:
        ego_df: Ego trajectory DataFrame with 'velocity' column.

    Returns:
        Average speed in m/s.
    """
    return ego_df['velocity'].mean()


def _progress_rate(ego_df: pd.DataFrame, duration_s: float) -> float:
    """Straight-line displacement divided by elapsed time.

    Measures how effectively the vehicle moves toward its destination.
    A vehicle that drives in circles has high average_speed but low
    progress_rate.

    Args:
        ego_df: Ego trajectory DataFrame with 'x' and 'y' columns.
        duration_s: Episode duration in seconds.

    Returns:
        Progress rate in m/s.
    """
    if duration_s < 0.01:
        return 0.0
    dx = ego_df["x"].iloc[-1] - ego_df["x"].iloc[0]
    dy = ego_df["y"].iloc[-1] - ego_df["y"].iloc[0]
    displacement = np.sqrt(dx ** 2 + dy ** 2)
    return displacement / duration_s


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_metrics(
    episode_id: str,
    ego_df: pd.DataFrame,
    agents_df: pd.DataFrame,
    duration_s: float,
) -> EpisodeMetrics:
    """Compute all metrics for a single driving episode.

    This is the main entry point for the metric engine.

    Args:
        episode_id: Unique ID for this episode.
        ego_df: Ego vehicle trajectory DataFrame (from extract.py).
        agents_df: Other agents trajectory DataFrame (from extract.py).
        duration_s: Episode duration in seconds.

    Returns:
        EpisodeMetrics dataclass with all computed values.
    """
    return EpisodeMetrics(
        episode_id=episode_id,
        max_jerk=_max_jerk(ego_df),
        rms_jerk=_rms_jerk(ego_df),
        max_lateral_accel=_max_lateral_accel(ego_df),
        rms_lateral_accel=_rms_lateral_accel(ego_df),
        braking_smoothness=_braking_smoothness(ego_df),
        speed_consistency=_speed_consistency(ego_df),
        min_ttc=_min_ttc(ego_df, agents_df),
        min_agent_distance=_min_agent_distance(ego_df, agents_df),
        hard_braking_count=_hard_braking_count(ego_df),
        hard_braking_rate=_hard_braking_rate(ego_df, duration_s),
        average_speed=_average_speed(ego_df),
        progress_rate=_progress_rate(ego_df, duration_s),
    )


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from extract import extract_single_file
    import sys

    if len(sys.argv) < 2:
        print("Usage: python metrics.py <path_to_tfrecord_file>")
        sys.exit(1)

    episodes = extract_single_file(sys.argv[1])
    if not episodes:
        print("No episodes extracted.")
        sys.exit(1)

    # Compute metrics for first episode as a test
    ep = episodes[0]
    metrics = compute_metrics(ep.episode_id, ep.ego_df, ep.agents_df, ep.duration_s)
    print(f"\nMetrics for episode {metrics.episode_id}:")
    for key, value in metrics.to_dict().items():
        if key == "episode_id":
            continue
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
