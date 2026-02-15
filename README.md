# Financial Intelligence AI Agent - Consumer Complaints Analysis

A comprehensive data pipeline to ingest CFPB consumer complaints data into Google Cloud, perform exploratory data analysis, and prepare data for fine-tuning a Gemma-3 model using QLoRA.

## Overview

```
complaints.csv.gz (local)
     ↓
setup_infra.py (create infrastructure)
     ↓ GCS buckets + BigQuery dataset
load_raw_data.py (ingest to raw bucket)
     ↓ gs://cfpb-usd-aai-540-cfpb-raw-lake/raw/complaints.csv.gz
preprocess.py (transform & load to BigQuery)
     ↓ cfpb_analysis.stg_complaints table
eda_cfpb_dataset.py (exploratory analysis)
     ↓ Query-based insights and statistics
```

## Prerequisites

### 1. Environment Setup

```bash
# Install dependencies
pip install -r setup/requirement.txt

# Set Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Authenticate with gcloud (alternative)
gcloud auth application-default login
```

### 2. Required Files

Ensure the complaints data exists:
```bash
ls -lh data/complaints.csv.gz
```

### 3. Configuration

All settings are in [config/config.yaml](config/config.yaml):
- **Project ID:** `cfpb-usd-aai-540`
- **Region:** `us-central1`
- **Raw Bucket:** `cfpb-usd-aai-540-cfpb-raw-lake`
- **BigQuery Dataset:** `cfpb_analysis`
- **Raw File Path:** `data/complaints.csv.gz`

## Pipeline Execution

### Step 1: Setup Infrastructure

Creates GCS buckets and BigQuery dataset:

```bash
python src/ingestion/setup_infra.py
```

**Output:**
```
--- Setting up infrastructure for Project: cfpb-usd-aai-540 in us-central1 ---
Created GCS Bucket: cfpb-usd-aai-540-cfpb-raw-lake
GCS Bucket already exists: cfpb-usd-aai-540-cfpb-clean-features
GCS Bucket already exists: cfpb-usd-aai-540-cfpb-weights-registry
Created BigQuery Dataset: cfpb-usd-aai-540.cfpb_analysis
--- Infrastructure setup complete. ---
```

### Step 2: Ingest Raw Data

Uploads the gzip file to GCS raw bucket:

```bash
python src/ingestion/load_raw_data.py
```

**Output:**
```
Uploading ./data/complaints.csv.gz to gs://cfpb-usd-aai-540-cfpb-raw-lake/raw/complaints.csv.gz...
   File size: 45.23 MB
Success! File available at: gs://cfpb-usd-aai-540-cfpb-raw-lake/raw/complaints.csv.gz
```

**Verify in GCS:**
```bash
gsutil ls -lh gs://cfpb-usd-aai-540-cfpb-raw-lake/raw/
```

### Step 3: Preprocess & Load to BigQuery

Reads gzip from GCS and creates staging table:

```bash
python src/preprocessing/preprocess.py
```

**Output:**
```
Attempting to read from: gs://cfpb-usd-aai-540-cfpb-raw-lake/raw/complaints.csv.gz
Found file: gs://cfpb-usd-aai-540-cfpb-raw-lake/raw/complaints.csv.gz
GCS RAW File Location: gs://cfpb-usd-aai-540-cfpb-raw-lake/raw/complaints.csv.gz
Reading data from GCS...
Total rows loaded: 500,000
Performing preprocessing...
Writing to BigQuery staging table: cfpb-usd-aai-540.cfpb_analysis.stg_complaints
Preprocessing complete. Staging table created: cfpb-usd-aai-540.cfpb_analysis.stg_complaints
```

**Verify in BigQuery:**
```bash
bq query --use_legacy_sql=false 'SELECT COUNT(*) as total_rows FROM `cfpb-usd-aai-540.cfpb_analysis.stg_complaints`'
```

### Step 4: Run Exploratory Data Analysis

Performs comprehensive EDA on BigQuery data:

```bash
python src/eda/eda_cfpb_dataset.py
```

**Analysis includes:**
- **Table Overview:** Total rows, date range (min/max)
- **Null Analysis:** Counts null values across key fields
- **Temporal Distribution:** Complaints grouped by year
- **Label Distribution:** Top 20 products by complaint count
- **Narrative Length Analysis:** Decile-based text length statistics
- **Raw File Validation:** Schema validation on local gzip (if available)

**Output:**
```
=== CFPB Consumer Complaints EDA ===
Run time: 2026-02-15T12:34:56.789Z

[Phase 1] Raw file schema validation
Column count: 18
Column names:
  - date_received
  - product
  - issue
  - complaint_what_happened
  ...

[Phase 2] BigQuery table overview
   total_rows  min_date  max_date
0      500000 2011-06-01 2024-12-31

[Phase 2] Null analysis
  issue_nulls  company_nulls  product_nulls  narrative_nulls
0         245         1203        305           12547

[Phase 2] Temporal distribution
    year  complaints
0   2011       2345
1   2012       3456
...

[Phase 2] Label distribution (products)
        product     count
0   Credit card    125000
1   Bank account    95000
...

EDA complete
```

## Project Structure

```
aai-540-in1-fin-int-ai-agent/
├── config/
│   └── config.yaml                 # Project configuration
├── data/
│   └── complaints.csv.gz           # Input data file
├── setup/
│   ├── environment.yml             # Conda environment
│   └── requirement.txt             # Python dependencies
├── src/
│   ├── eda/
│   │   └── eda_cfpb_dataset.py    # Exploratory data analysis
│   ├── ingestion/
│   │   ├── setup_infra.py         # Create GCS buckets & BQ dataset
│   │   └── load_raw_data.py       # Upload gzip to GCS
│   ├── preprocessing/
│   │   └── preprocess.py          # Transform raw → BQ staging
│   └── utils/
│       └── config_loader.py       # Configuration loader
├── main.ipynb                      # Jupyter notebook orchestration
└── README.md                       # This file
```

## Key Components

### [src/ingestion/setup_infra.py](src/ingestion/setup_infra.py)
- Creates GCS buckets (raw, clean, weights)
- Creates BigQuery dataset
- Handles credentials and permissions
- Safe (skips existing resources)

### [src/ingestion/load_raw_data.py](src/ingestion/load_raw_data.py)
- Uploads local gzip to GCS raw bucket
- Supports both `.csv.gz` and `.csv` formats
- Auto-detects file path from config
- Reports file size and status

### [src/preprocessing/preprocess.py](src/preprocessing/preprocess.py)
- Reads gzip from GCS using BigFrames
- Transforms data (extensible for custom logic)
- Materializes to BigQuery staging table
- Auto-fallback for multiple file formats

### [src/eda/eda_cfpb_dataset.py](src/eda/eda_cfpb_dataset.py)
- **Phase 1:** Validates raw gzip schema and structure
- **Phase 2:** Comprehensive BigQuery analysis
  - Table statistics
  - Data quality metrics
  - Temporal and categorical distributions
  - Text length analysis

### [src/utils/config_loader.py](src/utils/config_loader.py)
- Singleton pattern for thread-safe config access
- YAML-based configuration
- Environment variable override support
- Path resolution relative to project root

## Troubleshooting

### "File not found" in load_raw_data.py
```bash
# Check if file exists
ls -la data/complaints.csv.gz

# If missing, download or copy the file to data/ directory
```

### "Bucket not found" or "Permission denied"
```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Verify credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
```

### BigQuery table not created in preprocess.py
```bash
# Verify dataset exists
bq ls -d cfpb-usd-aai-540:cfpb_analysis

# Re-run infrastructure setup if needed
python src/ingestion/setup_infra.py
```

### GCS connection issues
```bash
# Test GCS access
gsutil ls gs://cfpb-usd-aai-540-cfpb-raw-lake/

# Check bucket permissions
gsutil iam ch -d user:your-email@example.com gs://cfpb-usd-aai-540-cfpb-raw-lake/
```

## Environment Variables

```bash
# Google Cloud Project
export GOOGLE_CLOUD_PROJECT=cfpb-usd-aai-540

# Service Account Credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# (Optional) BigFrames location
export BQ_LOCATION=us-central1
```

## Next Steps

1. **Extend Preprocessing:** Customize column selection, null handling, and feature engineering in [src/preprocessing/preprocess.py](src/preprocessing/preprocess.py)

2. **Feature Engineering:** Create derived features using BigQuery SQL

3. **Model Training:** Use the cleaned data for fine-tuning Gemma-3 with QLoRA

4. **Monitoring:** Set up data quality checks and drift detection

5. **BigQuery Analysis:** Explore data with custom SQL queries:
   ```sql
   SELECT product, COUNT(*) as count
   FROM `cfpb-usd-aai-540.cfpb_analysis.stg_complaints`
   GROUP BY product
   ORDER BY count DESC
   LIMIT 20
   ```

## Configuration Reference

[config/config.yaml](config/config.yaml) settings:

| Setting | Value | Description |
|---------|-------|-------------|
| `project.id` | `cfpb-usd-aai-540` | Google Cloud Project ID |
| `project.region` | `us-central1` | Default region |
| `data.raw_bucket` | `cfpb-raw-lake` | GCS raw bucket suffix |
| `data.raw_file_path` | `data/complaints.csv.gz` | Local file path |
| `bigquery.dataset` | `cfpb_analysis` | BigQuery dataset |
| `bigquery.tables.complaints_raw` | `stg_complaints` | Staging table name |

## License

This project is part of AAI-540 Financial Intelligence capstone.