"""
EDA Pipeline for CFPB Consumer Complaints
-----------------------------------------

Phase 1: Raw CSV / GZIP sanity checks (schema, encoding, corruption)
Phase 2: BigQuery-based exploratory data analysis (canonical truth)

Author: Financial Intelligence System
"""

import gzip
import csv
import sys
import os
from collections import Counter
from datetime import datetime, timezone
from typing import List

import pandas as pd
from google.cloud import bigquery


# =========================
# CONFIGURATION
# =========================

RAW_GZIP_PATH = "../../data/complaints.csv.gz"  # local path or downloaded sample
# Ensure project `src` is on PYTHONPATH so local imports work when running this file directly
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils import config_loader as cl

BQ_PROJECT = cl.config_loader.get(cl.PROJECT_ID)
BQ_DATASET = cl.config_loader.get(cl.BQ_DATASET) or "cfpb_analysis"
BQ_TABLE = cl.config_loader.get("bigquery.tables.complaints_raw") or "stg_complaints"
BQ_LOCATION = "us-central1"

RAW_SAMPLE_ROWS = 50_000  # safe sampling size


# =========================
# PHASE 1: RAW FILE EDA
# =========================

def raw_file_schema_check(path: str, sample_rows: int = 10_000):
    """
    Validate CSV structure without loading full file into memory.
    """
    print("\n[Phase 1] Raw file schema validation")

    with gzip.open(path, mode="rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)

        print(f"Column count: {len(header)}")
        print("Column names:")
        for col in header:
            print(f"  - {col}")

        row_lengths = Counter()
        for i, row in enumerate(reader):
            row_lengths[len(row)] += 1
            if i >= sample_rows:
                break

    print("\nRow length distribution (sampled):")
    for k, v in row_lengths.items():
        print(f"  {k} columns → {v} rows")

    if len(row_lengths) > 1:
        print("WARNING: Inconsistent row lengths detected")


def raw_file_content_probe(path: str, sample_rows: int = 50_000):
    """
    Lightweight content EDA using pandas sampling.
    """
    print("\n[Phase 1] Raw content probe (pandas sample)")

    df = pd.read_csv(
        path,
        compression="gzip",
        nrows=sample_rows,
        encoding="utf-8",
        on_bad_lines="skip"
    )

    print(df.info())
    print("\nNull percentage (top fields):")
    print((df.isna().mean() * 100).sort_values(ascending=False).head(10))

    if "Consumer complaint narrative" in df.columns:
        lengths = df["Consumer complaint narrative"].dropna().str.len()
        print("\nNarrative length stats:")
        print(lengths.describe())


# =========================
# PHASE 2: BIGQUERY EDA
# =========================

def bq_client():
    return bigquery.Client(project=BQ_PROJECT)


def run_bq_query(sql: str):
    client = bq_client()
    job = client.query(sql)
    return job.result().to_dataframe()


def bq_table_overview():
    print("\n[Phase 2] BigQuery table overview")

    sql = f"""
    SELECT
      COUNT(*) AS total_rows,
      MIN(`Date received`) AS min_date,
      MAX(`Date received`) AS max_date
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    """
    print(run_bq_query(sql))


def bq_null_analysis():
    print("\n[Phase 2] Null analysis")

    sql = f"""
    SELECT
      COUNTIF(`Issue` IS NULL) AS issue_nulls,
      COUNTIF(`Company` IS NULL) AS company_nulls,
      COUNTIF(`Product` IS NULL) AS product_nulls,
      COUNTIF(`Consumer complaint narrative` IS NULL) AS narrative_nulls
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    """
    print(run_bq_query(sql))


def bq_temporal_distribution():
    print("\n[Phase 2] Temporal distribution")

    sql = f"""
    SELECT
      EXTRACT(YEAR FROM `Date received`) AS year,
      COUNT(*) AS complaints
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    GROUP BY year
    ORDER BY year
    """
    print(run_bq_query(sql))


def bq_label_distribution(limit: int = 20):
    print("\n[Phase 2] Label distribution (products)")

    sql = f"""
    SELECT
      `Product`,
      COUNT(*) AS count
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    GROUP BY `Product`
    ORDER BY count DESC
    LIMIT {limit}
    """
    print(run_bq_query(sql))


def bq_narrative_length_analysis():
    print("\n[Phase 2] Narrative length distribution")

    sql = f"""
    SELECT
      APPROX_QUANTILES(LENGTH(`Consumer complaint narrative`), 10) AS length_deciles
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE `Consumer complaint narrative` IS NOT NULL
    """
    print(run_bq_query(sql))


# =========================
# MAIN
# =========================

def main():
    print("\n=== CFPB Consumer Complaints EDA ===")
    print(f"Run time: {datetime.now(timezone.utc).isoformat()}Z")

    # Phase 1
    try:
        raw_file_schema_check(RAW_GZIP_PATH)
        raw_file_content_probe(RAW_GZIP_PATH)
    except FileNotFoundError:
        print("Raw file not found. Skipping Phase 1.")

    # Phase 2
    bq_table_overview()
    bq_null_analysis()
    bq_temporal_distribution()
    bq_label_distribution()
    bq_narrative_length_analysis()

    print("\nEDA complete")


if __name__ == "__main__":
    main()
