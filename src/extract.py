"""
Data Ingestion Module — extract.py

Parses Waymo Open Dataset Motion TFRecord files into structured
pandas DataFrames ready for metric computation.

Waymo Motion Dataset proto structure (scenario.proto):
    Scenario
    ├── scenario_id: str
    ├── timestamps_seconds: List[float]   (length T, sampled at 10Hz)
    ├── sdc_track_index: int              (which track is the ego vehicle)
    ├── tracks: List[Track]               (all agents in the scene)
    │   └── Track
    │       ├── id: int
    │       ├── object_type: enum         (VEHICLE=1, PEDESTRIAN=2, CYCLIST=3, OTHER=4)
    │       └── states: List[ObjectState] (length T, one per timestamp)
    │           └── ObjectState
    │               ├── center_x, center_y, center_z: float (meters, world frame)
    │               ├── heading: float    (radians)
    │               ├── velocity_x, velocity_y: float (m/s, world frame)
    │               ├── length, width, height: float (meters)
    │               └── valid: bool       (whether this state is observed)
    └── map_features: List[MapFeature]    (lanes, boundaries, crosswalks, etc.)
"""

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import struct

import numpy as np
import pandas as pd

# Waymo proto import — we compile from their .proto or parse raw features.
# The scenario proto is stored as serialized bytes inside TFRecords.
# We need their compiled proto to deserialize. Install via:
#   pip install waymo-open-dataset-tf-2-12-0  (Linux only)
# On Windows, we'll parse manually using tf.train.Example or grab the
# proto from their GitHub and compile it ourselves.
#
# For now, we use the proto approach:
try:
    import waymo_scenario_pb2 as scenario_pb2
    HAS_WAYMO_PROTO = True
except ImportError:
    try:
        from src import waymo_scenario_pb2 as scenario_pb2
        HAS_WAYMO_PROTO = True
    except ImportError:
        HAS_WAYMO_PROTO = False
        print(
            "WARNING: Compiled proto not found. "
            "Run: python -m grpc_tools.protoc --proto_path=protos "
            "--python_out=src protos/waymo_scenario.proto"
        )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

OBJECT_TYPE_MAP = {
    0: "unset",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}


@dataclass
class EpisodeData:
    """Container for a single parsed driving episode."""
    episode_id: str
    ego_df: pd.DataFrame       # Ego vehicle trajectory
    agents_df: pd.DataFrame    # All other agents' trajectories
    duration_s: float          # Total episode duration in seconds
    num_agents: int            # Number of other agents in scene


# ---------------------------------------------------------------------------
# TFRecord reading
# ---------------------------------------------------------------------------

def list_tfrecord_files(data_dir: str) -> List[str]:
    """Find all TFRecord files in the given directory.

    Args:
        data_dir: Path to directory containing .tfrecord files.

    Returns:
        Sorted list of absolute file paths.
    """
    # Construct the full pattern
    search_path = os.path.join(data_dir, '*tfrecord*')
    
    files = glob.glob(search_path)
    print(f"Found {len(files)} files.")
    return sorted(files)


def read_tfrecord_file(filepath: str) -> List[bytes]:
    """Read all serialized scenario protos from a single TFRecord file.

    Each TFRecord file contains multiple serialized Scenario protocol buffers.
    Each scenario is one 20-second driving episode.

    Args:
        filepath: Path to a single TFRecord file.

    Returns:
        List of serialized proto bytes, one per scenario in the file.
    """
    # TFRecord format: each record is stored as:
    #   uint64 length
    #   uint32 masked_crc32_of_length
    #   byte   data[length]
    #   uint32 masked_crc32_of_data
    records = []
    with open(filepath, "rb") as f:
        while True:
            # Read 8-byte little-endian length
            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                break
            length = struct.unpack("<Q", length_bytes)[0]
            # Skip length CRC (4 bytes)
            f.read(4)
            # Read the actual data
            data = f.read(length)
            # Skip data CRC (4 bytes)
            f.read(4)
            records.append(data)
    print(f"  Read {len(records)} scenarios from {os.path.basename(filepath)}")
    return records


# ---------------------------------------------------------------------------
# Proto parsing
# ---------------------------------------------------------------------------

def parse_scenario(serialized: bytes) -> Optional[EpisodeData]:
    """Parse a single serialized Scenario proto into an EpisodeData.

    This is the main parsing function. It:
    1. Deserializes the proto
    2. Extracts the ego vehicle track (identified by sdc_track_index)
    3. Extracts all other agent tracks
    4. Builds DataFrames with derived signals

    Args:
        serialized: Raw bytes of a serialized Scenario proto.

    Returns:
        EpisodeData if parsing succeeds, None if the scenario is invalid.
    """
    if not HAS_WAYMO_PROTO:
        # Implement manual parsing without the Waymo proto
        # For now, require the proto package
        raise RuntimeError("Waymo proto package required. See README.")

    # Step 1: Deserialize the proto
    # Create a Scenario proto object and parse from serialized bytes
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(serialized)

    # Step 2: Extract timestamps
    # Get the list of timestamps from scenario.timestamps_seconds
    # Convert to numpy array
    timestamps = np.array(scenario.timestamps_seconds)

    # Step 3: Identify the ego vehicle track
    # Use scenario.sdc_track_index to find the ego tracks in scenario.tracks
    ego_track = scenario.tracks[scenario.sdc_track_index]
    
    # Step 4: Parse ego track into DataFrame
    # Call _track_to_dataframe() for the ego track and add episode_id column from scenario.scenario_id
    ego_df = _track_to_dataframe(ego_track, timestamps)
    ego_df["episode_id"] = scenario.scenario_id

    # Step 5: Parse all other agent tracks into a single DataFrame
    # Loop over all tracks except the ego track
    #       Call _track_to_dataframe() for each
    #       Add agent_id, agent_type, episode_id columns
    #       Concatenate into one DataFrame
    agent_dfs = []
    for i, track in enumerate(scenario.tracks):
        if i == scenario.sdc_track_index:
            continue
        
        agent_df = _track_to_dataframe(track, timestamps)
        if agent_df.empty:
            continue
        
        agent_df["agent_id"] = track.id
        agent_df["agent_type"] = OBJECT_TYPE_MAP.get(track.object_type, "unknown")
        agent_df["episode_id"] = scenario.scenario_id
        agent_dfs.append(agent_df)
        
    agents_df = pd.concat(agent_dfs, ignore_index=True) if agent_dfs else pd.DataFrame()
        

    # Step 6: Compute derived signals on the ego DataFrame
    # Call _compute_derived_signals() on ego_df
    ego_df = _compute_derived_signals(ego_df)
    
    # Step 7: Build and return EpisodeData
    return EpisodeData(
        episode_id=scenario.scenario_id,
        ego_df=ego_df,
        agents_df=agents_df,
        duration_s=timestamps[-1] - timestamps[0],
        num_agents=len(agent_dfs),
    )


def _track_to_dataframe(
    track,  # scenario_pb2.Track proto object
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """Convert a single Track proto into a pandas DataFrame.

    Only includes timesteps where the state is valid (state.valid == True).

    Args:
        track: A Track proto containing repeated ObjectState.
        timestamps: Array of timestamps corresponding to each state index.

    Returns:
        DataFrame with columns:
            timestamp, x, y, heading, velocity_x, velocity_y, length, width
    """
    records = []
    for state, ts in zip(track.states, timestamps):
        if not state.valid:
            continue
        records.append({
            "timestamp": ts,
            "x": state.center_x,
            "y": state.center_y,
            "heading": state.heading,
            "velocity_x": state.velocity_x,
            "velocity_y": state.velocity_y,
            "length": state.length,
            "width": state.width,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------

def _compute_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived kinematic signals to an ego trajectory DataFrame.

    Computes signals that aren't directly in the proto but are needed
    for metric computation. All derivatives use finite differences.

    Args:
        df: DataFrame with columns [timestamp, x, y, heading,
            velocity_x, velocity_y]

    Returns:
        Same DataFrame with added columns:
            velocity     — scalar speed (m/s)
            acceleration — longitudinal acceleration (m/s²)
            yaw_rate     — rate of heading change (rad/s)
            jerk         — derivative of acceleration (m/s³)
            lateral_acceleration — velocity * yaw_rate (m/s²)
    """
    # Scalar speed from velocity components
    df["velocity"] = np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2)

    # Time delta between consecutive rows (should be ~0.1s at 10Hz)
    dt = df["timestamp"].diff()

    # Longitudinal acceleration via finite difference of speed
    df["acceleration"] = df["velocity"].diff() / dt

    # Yaw rate via finite difference of heading.
    # Heading wraps at ±pi, so unwrap first to avoid false spikes
    # (e.g., going from +3.1 to -3.1 is a tiny turn, not a 6.2 rad jump)
    unwrapped_heading = np.unwrap(df["heading"].values)
    df["yaw_rate"] = pd.Series(unwrapped_heading).diff().values / dt.values

    # Jerk = derivative of acceleration (how abruptly acceleration changes)
    df["jerk"] = df["acceleration"].diff() / dt

    # Lateral acceleration = centripetal force felt by passengers in turns
    df["lateral_acceleration"] = df["velocity"] * df["yaw_rate"]

    # First 1-2 rows are NaN from diff(). Fill with 0 to preserve row count
    # so ego_df and agents_df stay aligned on timestamps.
    df.fillna(0, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_episode(episode: EpisodeData) -> Tuple[bool, List[str]]:
    """Check an episode for data quality issues.

    Flags episodes with:
    - Too few valid timesteps (< 50% of expected)
    - Physically impossible values (acceleration > 15 m/s², speed > 50 m/s)
    - Large gaps in timestamps
    - NaN values in critical columns

    Args:
        episode: A parsed EpisodeData object.

    Returns:
        Tuple of (is_valid, list_of_warnings).
    """
    warnings = []
    ego = episode.ego_df

    # Check minimum number of timesteps
    if len(ego) < 100:
        warnings.append(f"Too few timesteps: {len(ego)} (expected ~200)")

    # Check for physically impossible acceleration
    if "acceleration" in ego.columns:
        max_accel = ego["acceleration"].abs().max()
        if max_accel > 15.0:
            warnings.append(f"Suspect acceleration: {max_accel:.1f} m/s² (threshold 15)")

    # Check for physically impossible speed
    if "velocity" in ego.columns:
        max_speed = ego["velocity"].max()
        if max_speed > 50.0:
            warnings.append(f"Suspect speed: {max_speed:.1f} m/s (threshold 50)")

    # Check for timestamp gaps
    dt = ego["timestamp"].diff().dropna()
    max_gap = dt.max()
    if max_gap > 0.5:
        warnings.append(f"Large timestamp gap: {max_gap:.2f}s (threshold 0.5)")

    # Check for NaN values in critical columns
    critical_cols = ["x", "y", "velocity", "acceleration"]
    for col in critical_cols:
        if col in ego.columns:
            nan_count = ego[col].isna().sum()
            if nan_count > 0:
                warnings.append(f"NaN values in {col}: {nan_count}")

    is_valid = len(warnings) == 0
    return is_valid, warnings


# ---------------------------------------------------------------------------
# High-level pipeline functions
# ---------------------------------------------------------------------------

def extract_all_episodes(data_dir: str) -> List[EpisodeData]:
    """Parse all TFRecord files in a directory into EpisodeData objects.

    This is the main entry point for the extraction pipeline.

    Args:
        data_dir: Path to directory containing TFRecord files.

    Returns:
        List of validated EpisodeData objects.
    """
    files = list_tfrecord_files(data_dir)
    if not files:
        print("No TFRecord files found.")
        return []

    valid_episodes = []
    total_scenarios = 0
    skipped = 0

    for filepath in files:
        records = read_tfrecord_file(filepath)
        for serialized in records:
            total_scenarios += 1
            episode = parse_scenario(serialized)
            if episode is None:
                skipped += 1
                continue

            is_valid, warnings = validate_episode(episode)
            if is_valid:
                valid_episodes.append(episode)
            else:
                skipped += 1
                print(f"  Skipping {episode.episode_id}: {warnings}")

    print(f"\nSummary: {len(files)} files, {total_scenarios} scenarios, "
          f"{len(valid_episodes)} valid, {skipped} skipped")
    return valid_episodes


def extract_single_file(filepath: str) -> List[EpisodeData]:
    """Parse a single TFRecord file. Useful for testing/debugging.

    Args:
        filepath: Path to a single TFRecord file.

    Returns:
        List of EpisodeData objects from that file.
    """
    records = read_tfrecord_file(filepath)
    episodes = []
    for serialized in records:
        episode = parse_scenario(serialized)
        if episode is None:
            continue
        is_valid, warnings = validate_episode(episode)
        if is_valid:
            episodes.append(episode)
        else:
            print(f"  Skipping {episode.episode_id}: {warnings}")
    return episodes


# ---------------------------------------------------------------------------
# Entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract.py <path_to_data_dir_or_single_file>")
        print("Example: python extract.py data/raw/")
        print("Example: python extract.py data/raw/training_20s.tfrecord-00000-of-01000")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        episodes = extract_all_episodes(path)
    else:
        episodes = extract_single_file(path)

    print(f"\nExtracted {len(episodes)} valid episodes.")

    if episodes:
        sample = episodes[0]
        print(f"\nSample episode: {sample.episode_id}")
        print(f"  Duration: {sample.duration_s:.1f}s")
        print(f"  Ego rows: {len(sample.ego_df)}")
        print(f"  Agents: {sample.num_agents}")
        print(f"\nEgo DataFrame columns: {list(sample.ego_df.columns)}")
        print(f"\nEgo DataFrame head:")
        print(sample.ego_df.head(10).to_string())
