# AV Planner Evaluation Pipeline

A data pipeline that processes autonomous driving logs from the [Waymo Open Dataset](https://waymo.com/open/), computes driving quality metrics, and produces statistical analysis comparing driving episodes.

Built to mirror the core work of an AV planner evaluation team: measuring and improving the quality of the software that drives the car.

## Architecture

```
Waymo Open Dataset (TFRecord files)
        |
        v
+---------------------+
|  1. Data Ingestion   |  Parse protobuf -> structured DataFrames
|     (extract.py)     |  Ego vehicle + other agents per episode
+---------+-----------+
          v
+---------------------+
|  2. Metric Engine    |  12 metrics across comfort, safety, efficiency
|     (metrics.py)     |  One scalar per metric per episode
+---------+-----------+
          v
+---------------------+
|  3. Storage          |  SQLite database for querying across episodes
|     (database.py)    |  Long and wide format views
+---------+-----------+
          v
+---------------------+
|  4. Analysis         |  Descriptive stats, hypothesis tests,
|     (analysis.py)    |  correlation analysis, outlier detection
+---------+-----------+
          v
+---------------------+
|  5. Reporting        |  HTML report with histograms, heatmaps,
|     (report.py)      |  tables, and recommendations
+---------------------+
```

## Metrics

### Comfort (ride quality)

| Metric | Unit | Description |
|--------|------|-------------|
| Max Jerk | m/s³ | Worst sudden acceleration change |
| RMS Jerk | m/s³ | Overall smoothness |
| Max Lateral Accel | m/s² | Sharpest turn force felt by rider |
| RMS Lateral Accel | m/s² | Overall turning comfort |
| Braking Smoothness | m/s² | Consistency of braking (std of deceleration) |
| Speed Consistency | ratio | Coefficient of variation of speed |

### Safety

| Metric | Unit | Description |
|--------|------|-------------|
| Min Time-to-Collision | s | Closest time-to-collision with any agent |
| Min Agent Distance | m | Closest physical proximity to any agent |
| Hard Braking Count | count | Emergency braking events (accel < -3.0 m/s²) |
| Hard Braking Rate | events/min | Frequency of hard braking |

### Efficiency

| Metric | Unit | Description |
|--------|------|-------------|
| Average Speed | m/s | Mean speed over the episode |
| Progress Rate | m/s | Straight-line displacement / time |

## Setup

### Prerequisites

- Python 3.9+
- Google Cloud SDK (for downloading data)

### Install dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn
pip install grpcio-tools   # for proto compilation
```

### Compile the proto

The Waymo dataset uses Protocol Buffers. Compile the proto to generate the Python parser:

```bash
python -m grpc_tools.protoc --proto_path=protos --python_out=src protos/waymo_scenario.proto
```

This generates `src/waymo_scenario_pb2.py` which is required for reading the data.

### Download the data

1. Go to https://waymo.com/open/ and accept the terms of use
2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   ```
3. Download a subset of the Motion Dataset (training_20s):
   ```bash
   gcloud storage cp "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training_20s/training_20s.tfrecord-0000*-of-01000" data/raw
   ```

Each shard is ~90 MB and contains ~75 driving episodes (20 seconds each at 10 Hz). 10-20 shards is sufficient for development.

## Usage

### Run the full pipeline

```bash
python src/extract.py data/raw/
```

### Test on a single file

```bash
python src/extract.py data/raw/training_20s.tfrecord-00000-of-01000
```

### Compute metrics

```bash
python src/metrics.py data/raw/training_20s.tfrecord-00000-of-01000
```

### Run statistical analysis (synthetic test data)

```bash
python src/analysis.py
```

### Generate HTML report (synthetic test data)

```bash
python src/report.py
```

Report is saved to `output/report.html`.

## Project Structure

```
av-planner-evaluation/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── protos/
│   ├── waymo_scenario.proto      # Combined Waymo proto (scenario + map)
│   ├── scenario.proto            # Original from Waymo GitHub
│   └── map.proto                 # Original from Waymo GitHub
├── src/
│   ├── __init__.py
│   ├── waymo_scenario_pb2.py     # Generated proto parser (compiled)
│   ├── extract.py                # Data ingestion & parsing
│   ├── metrics.py                # Metric computation engine
│   ├── database.py               # SQLite storage layer
│   ├── analysis.py               # Statistical analysis
│   └── report.py                 # HTML report generation
├── data/
│   └── raw/                      # Waymo TFRecord files (gitignored)
├── output/                       # Generated reports & database (gitignored)
├── notebooks/                    # Jupyter notebooks for exploration
└── tests/
    ├── test_extract.py
    ├── test_metrics.py
    └── test_analysis.py
```

## Key Design Decisions

**Why these three metric categories?**
Comfort, safety, and efficiency are the three axes that matter for autonomous driving. Comfort measures rider experience, safety measures harm avoidance, efficiency measures getting places. A good planner must balance all three.

**Why SQLite?**
Metrics need to be queryable across thousands of episodes with flexible filtering. SQL gives you that. In production this would be BigQuery or a columnar store, but the query patterns are the same.

**Why statistical tests with effect sizes?**
p-values tell you if a difference is statistically significant. Effect size (Cohen's d) tells you if it's practically meaningful. With enough data, any tiny difference becomes "significant" — effect size separates real improvements from noise.

**Why multiple testing correction?**
Testing 12 metrics simultaneously at alpha=0.05 means ~0.6 expected false positives. Bonferroni and Benjamini-Hochberg control for this. Without correction, you'd report phantom improvements.

**Why TFRecord parsing without TensorFlow?**
TFRecord is a simple binary format (length-prefixed records). A 15-line struct-based reader avoids the ~2 GB TensorFlow dependency on Windows.

## Data Source

[Waymo Open Dataset - Motion Dataset v1.3.1](https://waymo.com/open/)

- 103,354 segments, each 20 seconds at 10 Hz
- Ego vehicle and all tracked agents (vehicles, pedestrians, cyclists)
- HD map data (lanes, stop signs, crosswalks, traffic signals)
- Licensed under Waymo Dataset License Agreement
