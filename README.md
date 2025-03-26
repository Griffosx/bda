# Vessel Anomaly Detection System

This project implements a system for detecting anomalies in maritime vessel data. It analyzes AIS (Automatic Identification System) data to identify two types of anomalies:

1. **Vessel Movement Anomalies**: Suspicious vessel behaviors like excessive speed or large gaps in AIS transmissions
2. **Location Anomalies**: Different vessels reporting identical positions at the same time (which may indicate spoofing)

## Table of Contents

- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Running Tests](#running-tests)
- [How it Works](#how-it-works)
  - [Data Preprocessing](#data-preprocessing)
  - [Vessel Anomaly Detection](#vessel-anomaly-detection)
  - [Location Anomaly Detection](#location-anomaly-detection)
  - [Performance Optimization](#performance-optimization)

## Technologies Used

- **Python 3.x** - Core programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical operations
- **geopy** - Geographic distance calculations
- **multiprocessing** - Parallel processing
- **tqdm** - Progress bar visualization
- **pytest** - Testing framework
- **Poetry** (optional) - Dependency management
- **Task** (optional) - Task runner (via Taskfile.yml)

## Project Structure

```
├── src/
│   └── assignment_1/
│       ├── location_anomalies.py - Detects vessels at identical locations
│       ├── vessel_anomalies.py - Detects vessel movement anomalies
│       └── main.py - Application entry point
│   └── utils/
│       └── timer_wrapper.py - Performance timing utility
├── tests/
│   ├── test_preprocess_vessel_spatial_data.py
│   ├── test_preprocess_vessel_temporal_data.py
│   └── test_vessel_anomalies.py
├── assets/ - Data files location
├── Taskfile.yml - Task definitions
└── poetry.toml - Dependency configuration
```

## Installation

### Option 1: Using Poetry (Optional)

Poetry is a tool for dependency management in Python that makes it easier to manage project dependencies. It's optional for this project.

1. Install Poetry:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

### Option 2: Standard Python Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy geopy tqdm pytest
   ```

3. Set up Python path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src  # On Windows: set PYTHONPATH=%PYTHONPATH%;%cd%\src
   ```

## How to Run

### Option 1: Using Task (Optional)

Task is a task runner similar to Make but written in Go, providing a simpler way to define and run commands. It's optional for this project.

```bash
task assignment_1
```

### Option 2: Direct Python Execution

```bash
# Make sure PYTHONPATH includes the src directory
python src/assignment_1/main.py
```

This processes the vessel data file specified in `main.py` (by default `aisdk-2024-06-30.csv`) and outputs anomaly reports.

## Running Tests

### Option 1: Using Task

```bash
task test
```

### Option 2: Direct pytest Execution

```bash
pytest
```

## How it Works

### Data Preprocessing

1. **Coordinate Normalization**: The system normalizes latitude and longitude coordinates to standard ranges (-90° to 90° for latitude, -180° to 180° for longitude).

2. **Spatial Binning**: Vessel positions are grouped into spatial bins based on their coordinates:

   - Default bin size: 0.01° latitude/longitude (approximately 1km)
   - Points near bin boundaries are placed in multiple bins to ensure anomalies at boundaries aren't missed

3. **Temporal Binning**: Data is also grouped by time:
   - Default bin size: 1 minute
   - Points near time boundaries are duplicated in adjacent time bins

### Vessel Anomaly Detection

The system detects two types of vessel movement anomalies:

1. **Speed Anomalies**:

   - Calculates the speed between consecutive AIS transmissions for each vessel
   - Flags vessels exceeding the maximum expected speed (default: 50 mph)
   - Uses `geopy.distance` to calculate distances between coordinates

2. **AIS Gap Anomalies**:
   - Identifies unusually large time gaps between successive AIS transmissions
   - Default threshold: 1 hour maximum time gap
   - May indicate when vessels are deliberately turning off their AIS transponders

The code processes each vessel's data separately and can handle vessel data with millions of records efficiently.

### Location Anomaly Detection

Location anomalies occur when different vessels report the same position at the same time:

1. **Binning Strategy**:

   - Data is partitioned into spatiotemporal bins (lat-lon-time)
   - Only positions within the same bin are compared to improve efficiency

2. **Conflict Detection**:

   - Positions are rounded to a fixed precision (6 decimal places)
   - A position-time hash is created for efficient comparison
   - Vessels with identical position-time hashes are flagged as potential conflicts

3. **Conflict Reporting**:
   - Reports pairs of vessels with identical positions
   - Calculates the time difference and distance between conflicting reports
   - Sorts conflicts by distance for easier analysis

### Performance Optimization

The system incorporates several optimizations:

1. **Vectorized Operations**:

   - Uses pandas and numpy vectorized operations for faster data processing
   - Avoids slow Python loops where possible

2. **Parallel Processing**:

   - Implements both single-process and multi-process versions for flexibility
   - Multi-process implementation uses Python's multiprocessing module
   - Automatically utilizes available CPU cores (N-1 processes by default)

3. **Progress Tracking**:

   - Uses tqdm to display progress bars during long-running operations
   - Provides timing information via the `@timeit` decorator

4. **Memory Efficiency**:
   - Processes data in chunks based on spatiotemporal bins
   - Avoids loading the entire dataset into memory at once for each operation

The multi-process implementation significantly improves performance on large datasets, especially on multi-core systems.
