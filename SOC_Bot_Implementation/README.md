# SOC Bot for Security Alerts Triage

## Overview

This project is a proof-of-concept bot designed to automate the triage of security alerts in a Security Operations Center (SOC). It processes historical alert data from Excel files for model training and is structured to later support real-time alert ingestion (e.g., from QRadar). The design emphasizes modularity, scalability, and maintainability.

## Directory Structure

```
my_soc_bot/
├── data/                      # Sample/test data (e.g., Excel files with historical alerts)
├── docs/                      # Documentation and diagrams (e.g., bot architecture, WBS)
├── logs/                      # (Optional) Log files directory
├── src/                       # Source code for the bot
│   ├── __init__.py
│   ├── main.py              # Main entry point; sets up logging and orchestrates processing
│   ├── logger.py            # Centralized logging configuration for uniform log formatting
│   ├── config.py            # Global configuration (e.g., file paths, verbosity setting)
│   ├── preprocessing/       # Data cleaning and normalization routines
│   │   ├── __init__.py
│   │   └── preprocess.py    # Functions to process single alerts and bulk Excel data
│   ├── models/              # Future module for ML model training and evaluation
│   │   ├── __init__.py
│   │   └── model_training.py # Example training logic (e.g., false-positive detection)
│   └── realtime/            # Module for handling real-time alert ingestion (e.g., QRadar alerts)
│       ├── __init__.py
│       └── qradar_ingestion.py # Converts and preprocesses real-time JSON alerts
├── tests/                     # Unit and integration tests for various modules
│   └── test_preprocessing.py
├── requirements.txt           # Project dependencies
└── README.md                  # This documentation file
```

## Key Components

### `src/main.py`
- **Purpose:** Acts as the CLI entry point.
- **Details:**
  - Parses command-line options (e.g., `-v` for verbose logging).
  - Configures the root logger so all modules use the same logging level.
  - Loads data, initiates preprocessing, and eventually triggers model training.

### `src/logger.py`
- **Purpose:** Provides a `get_logger()` function for consistent logging.
- **Details:**
  - Ensures a uniform log format (timestamps, log level, module name, etc.) across the project.
  - Loggers inherit the configuration set in `main.py` (using the root logger or a global configuration).

### `src/config.py`
- **Purpose:** Stores global configuration values.
- **Details:**
  - Includes constants like default file paths and verbosity settings.
  - Can be extended as additional configuration items are needed.

### `src/preprocessing/preprocess.py`
- **Purpose:** Contains routines to clean and normalize alert data.
- **Details:**
  - Offers functions for processing both single alerts (for real-time use) and entire DataFrames (for Excel data).
  - Performs tasks such as lowercasing text, removing punctuation, and trimming whitespace.

### `src/models/model_training.py`
- **Purpose:** Placeholder for ML model training.
- **Details:**
  - Includes example code (using a Random Forest) for training a false-positive detector.
  - Uses the centralized logger instead of print statements for uniform logging.

### `src/realtime/qradar_ingestion.py`
- **Purpose:** Handles real-time alert ingestion from QRadar.
- **Details:**
  - Converts JSON alerts to the same format as the Excel data.
  - Calls the shared preprocessing functions to ensure consistency across historical and live data.

## Setup and Execution

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Bot:**

   ```bash
   python -m src.main --excel-path data/sample_alerts.xlsx -v
   ```

   The `-v` flag enables verbose (DEBUG) logging across all modules.

## Future Development

- **Enhanced Model Training:** Expand the training pipeline with additional ML algorithms for false-positive detection and alert classification.
- **Real-Time Processing:** Enhance the real-time ingestion module to handle live feeds from systems like QRadar.
- **Comprehensive Testing:** Extend the test suite in the `tests/` directory to improve coverage and reliability.