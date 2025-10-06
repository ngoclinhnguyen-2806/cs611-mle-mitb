# Medallion Architecture for Loan Default Prediction

## Overview
This project implements a production-ready data pipeline using the Medallion Architecture (Bronze → Silver → Gold) to prepare data for machine learning model training to predict loan defaults.

---

## Architecture Design

### Medallion Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      Source Data (CSV)                      │
│  lms_loan_daily.csv | features_clickstream.csv |            │
│  features_attributes.csv | features_financials.csv          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    BRONZE LAYER (Raw)                       │
│  • Raw data ingestion from source systems                   │
│  • No transformations                                       │
│  • Partitioned by snapshot_date                             │
│  • Format: CSV                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              SILVER LAYER (Cleaned & Validated)             │
│  • Data type enforcement                                    │
│  • Feature engineering (MOB, DPD)                           │
│  • Data quality checks                                      │
│  • Format: Parquet                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           GOLD LAYER (Analytics-Ready Features)             │
│  • Label Store: Binary default labels                       │
│  • Feature Store: Model-ready features                      │
│  • Business logic applied                                   │
│  • Format: Parquet                                          │
└─────────────────────────────────────────────────────────────┘
```

### Data Tables

| Table Name | Description | Source File |
|------------|-------------|-------------|
| `lms_loan_daily` | Loan transaction and repayment data | `lms_loan_daily.csv` |
| `features_clickstream` | Customer behavioral features (fe_1 to fe_20) | `feature_clickstream.csv` |
| `features_attributes` | Customer demographics and attributes | `features_attributes.csv` |
| `features_financials` | Financial metrics and credit information | `features_financials.csv` |

---

## Directory Structure

```
project/
├── data/                                    # Source CSV files
│   ├── lms_loan_daily.csv
│   ├── feature_clickstream.csv
│   ├── features_attributes.csv
│   └── features_financials.csv
│
├── datamart/                                # Medallion Architecture layers
│   ├── bronze/                              # Raw data layer
│   │   ├── lms_loan_daily/
│   │   │   ├── bronze_lms_loan_daily_2023_01_01.csv
│   │   │   ├── bronze_lms_loan_daily_2023_02_01.csv
│   │   │   └── ...
│   │   ├── features_clickstream/
│   │   │   └── bronze_features_clickstream_YYYY_MM_DD.csv
│   │   ├── features_attributes/
│   │   │   └── bronze_features_attributes_YYYY_MM_DD.csv
│   │   └── features_financials/
│   │       └── bronze_features_financials_YYYY_MM_DD.csv
│   │
│   ├── silver/                              # Cleaned/transformed layer
│   │   ├── loan_daily/
│   │   │   └── silver_loan_daily_YYYY_MM_DD.parquet
│   │   ├── features_clickstream/
│   │   ├── features_attributes/
│   │   └── features_financials/
│   │
│   └── gold/                                # Analytics-ready layer
│       ├── label_store/
│       │   └── gold_label_store_YYYY_MM_DD.parquet
│       └── feature_store/
│
├── utils/                                   # Processing logic
│   ├── data_processing_bronze_table.py
│   ├── data_processing_silver_table.py
│   └── data_processing_gold_table.py
│
├── main.py                                 # Batch pipeline runner
└── bronze_label_store.py                   # Incremental ingestion script
```

---

## Pipeline Processing Details

### Bronze Layer
**Purpose:** Raw data ingestion with no transformations

**Process:**
- Read CSV files from source
- Filter by snapshot_date
- Save to partitioned directories
- One partition per date per table

**Output Format:** CSV partitioned by snapshot_date

### Silver Layer

**Purpose:** Clean and normalize Bronze CSVs into analytics-ready Spark DataFrames, then write Parquet partitions.

**Transformations:**
1. **Loan (lms_loan_daily):** type enforcement → type enforcement → derived features: mob, installments_missed, first_missed_date, dpd.
**MOB (Month on Book):** Calculate loan age in months
**DPD (Days Past Due):** Calculate days since first missed payment
   - `installments_missed = CEIL(overdue_amt / due_amt)`
   - `first_missed_date = snapshot_date - installments_missed months`
   - `dpd = DATEDIFF(snapshot_date, first_missed_date)`
2. **Clickstream:** type enforcement across feature columns.
3. **Attributes:** drop optional PII (e.g., SSN, Name) → type enforcement.
4. **Financials:** type enforcement; normalize Payment_Behaviour (unexpected/garbage → "Unknown"); parse Credit_History_Age into Credit_History_Age_Year (float) and Credit_History_Age_Month (total months)

**Output Format:** Parquet partitioned by snapshot_date

### Gold Layer

#### Label Store
**Purpose:** Create binary default labels for model training

**Process:**
1. Filter loans at specific MOB (e.g., 6 months)
2. Apply default definition (e.g., DPD ≥ 30 days)
3. Create binary label (1 = default, 0 = non-default)
4. Track label definition (e.g., "30dpd_6mob")

**Schema:**
```
- loan_id: String
- Customer_ID: String
- label: Integer (0 or 1)
- label_def: String
- snapshot_date: Date
```

**Output Format:** Parquet

#### Feature Store
**Purpose:** Combine features for model training (directory created, processing logic to be implemented)

---

## Usage

### Running the Full Pipeline

**Process all dates (backfill):**
```bash
python main.py
```

This will:
1. Generate monthly dates from 2023-01-01 to 2024-12-01
2. Process Bronze layer for all 4 tables
3. Process Silver layer (loan transformations + feature directories)
4. Process Gold layer (label store + feature store directory)
5. Verify outputs and show summary

**Expected runtime:** 5-10 minutes

### Running Incremental Ingestion

**Process Bronze layer for a single date:**
```bash
# All tables
python bronze_label_store.py --snapshotdate "2023-01-01"

# Specific tables only
python bronze_label_store.py --snapshotdate "2023-01-01" --tables "lms_loan_daily,features_clickstream"
```

---

## Output Verification

After running `python main.py`, you should see:

```
================================================================================
VERIFYING RESULTS
================================================================================

📦 Bronze Layer Tables:
  ✓ lms_loan_daily: 24 partitions
  ✓ features_clickstream: 24 partitions
  ✓ features_attributes: 24 partitions
  ✓ features_financials: 24 partitions

🔧 Silver Layer Tables:
  ✓ loan_daily: 24 partitions - Sample partition row count: 530
  ✓ features_clickstream: 24 partitions - Sample partition row count: 8974
  ✓ features_attributes: 24 partitions - Sample partition row count: 530
  ✓ features_financials: 24 partitions - Sample partition row count: 530

✨ Gold Layer Stores:
  ✓ label_store: X rows
    Schema: loan_id, Customer_ID, label, label_def, snapshot_date
    Label distribution:
      label=0: X records
      label=1: X records
  ✓ feature_store: directory created (processing pending)
```

## Technologies Used

- **PySpark:** Distributed data processing
- **Pandas:** Data manipulation
- **Parquet:** Columnar storage format
- **Python 3.x:** Primary programming language

---

## Next Steps

1. **Gold Layer Feature Store:** Combine all features with point-in-time correctness
2. **Model Training Dataset:** Join feature store with label store for final ML-ready dataset
3. **Model Development:** Train machine learning models for loan default prediction