# --- Standard libraries ---
import os
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- PySpark core & types ---
from pyspark.sql import functions as F
from pyspark.sql.functions import col, regexp_replace, trim, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, DateType

# --- CLI utilities (if you use them elsewhere) ---
import argparse

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# BASIC VALIDATION (kept)
# =============================================================================
def validate_customer_ids(df, table_name):
    """
    Ensure Customer_ID exists; warn on nulls; log unique count.
    """
    if "Customer_ID" not in df.columns:
        error_msg = f"{table_name}: Customer_ID column not found"
        logger.error(error_msg)
        raise ValueError(error_msg)

    null_customers = df.filter(col("Customer_ID").isNull()).count()
    if null_customers > 0:
        logger.warning(f"{table_name}: {null_customers} rows with null Customer_ID")

    unique_customers = df.select("Customer_ID").distinct().count()
    logger.info(f"{table_name}: {unique_customers} unique customers")

# =============================================================================
# HELPERS
# =============================================================================
def load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark):
    """
    Load bronze feature table and apply common preprocessing.
    Returns Spark DataFrame or None if not found.
    """
    try:
        bronze_table_directory = os.path.join(bronze_directory, table_name + "/")
        partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-','_')}.csv"
        filepath = os.path.join(bronze_table_directory, partition_name)

        if not os.path.exists(filepath):
            logger.warning(f"Bronze file not found: {filepath}")
            return None

        df = spark.read.csv(filepath, header=True, inferSchema=True)
        logger.info(f"Loaded {table_name} from {filepath}: {df.count()} rows")

        # Drop unnamed index column if present
        if "" in df.columns:
            df = df.drop("")
        if "_c0" in df.columns:
            df = df.drop("_c0")

        # Basic sanity: id/date types
        validate_customer_ids(df, table_name)
        if "Customer_ID" in df.columns:
            df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
        if "snapshot_date" in df.columns:
            df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

        return df
    except Exception as e:
        logger.error(f"Error loading bronze table {table_name}: {str(e)}")
        raise


def save_silver_table(df, snapshot_date_str, silver_directory, table_name):
    """
    Save DataFrame to silver layer as parquet (overwrite).
    """
    try:
        table_silver_directory = os.path.join(silver_directory, table_name + "/")
        if not os.path.exists(table_silver_directory):
            os.makedirs(table_silver_directory)
            logger.info(f"Created directory: {table_silver_directory}")

        partition_name = f"silver_{table_name}_{snapshot_date_str.replace('-','_')}.parquet"
        filepath = os.path.join(table_silver_directory, partition_name)

        df.write.mode("overwrite").parquet(filepath)
        logger.info(f"Saved {table_name} to {filepath}")
    except Exception as e:
        logger.error(f"Error saving silver table {table_name}: {str(e)}")
        raise


def cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date"), numeric_threshold=0.9):
    """
    Auto-detect numeric-looking string columns, clean, and cast to Integer/Float.
    """
    try:
        string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
        candidates = [c for c in string_cols if c not in exclude]

        for c in candidates:
            cleaned = regexp_replace(col(c), r"[^0-9\.\-]+", "")
            cleaned = trim(cleaned)

            cast_col = when(F.length(cleaned) == 0, None).otherwise(cleaned).cast(DoubleType())
            tmp = f"__num_{c}"

            ratio = df.withColumn(tmp, cast_col) \
                      .select((F.count(tmp) / F.count(F.lit(1))).alias("ratio")) \
                      .collect()[0]["ratio"]

            if ratio is not None and ratio >= numeric_threshold:
                df_with = df.withColumn(tmp, cast_col)
                max_frac = df_with.select(
                    F.max(F.abs(col(tmp) - F.floor(col(tmp)))).alias("max_frac")
                ).collect()[0]["max_frac"]

                is_integer = (max_frac is None) or (float(max_frac) == 0.0)
                target_type = IntegerType() if is_integer else FloatType()

                df = df_with.drop(c).withColumn(c, col(tmp).cast(target_type)).drop(tmp)
                logger.info(
                    "Auto-cast '%s' -> %s (cleaned non-numeric chars)",
                    c, "Integer" if is_integer else "Float"
                )
            else:
                logger.info(
                    "Kept '%s' as string (only %.2f%% numeric after cleaning)",
                    c, (ratio or 0.0) * 100
                )

        return df
    except Exception as e:
        logger.error(f"Error in transformation: {str(e)}")
        raise

# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def transform_clickstream(df):
    """
    Clickstream: apply generic numeric cleaning/casting across columns.
    """
    try:
        # sweep numeric-like strings (keep id/date as-is)
        df = cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date"))
        logger.info("Clickstream transformations: cast_to_numeric applied.")
        return df
    except Exception as e:
        logger.error(f"Error in clickstream transformation: {str(e)}")
        raise


def transform_attributes(df):
    """
    Attributes: drop any PII if present, then numeric sweep.
    """
    try:
        # optional PII pruning first
        pii_cols = [c for c in ["SSN", "Name"] if c in df.columns]
        if pii_cols:
            df = df.drop(*pii_cols)
            logger.info("Attributes: dropped PII columns %s", pii_cols)

        # sweep numeric-like strings (keep id/date as-is)
        df = cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date", "Occupation"))
        logger.info("Attributes transformations: cast_to_numeric applied.")
        return df
    except Exception as e:
        logger.error(f"Error in attributes transformation: {str(e)}")
        raise


def transform_financials(df):
    """
    Financials:
      1) numeric sweep via cast_to_numeric()
      2) normalize Payment_Behaviour: garbage/unexpected -> 'Unknown'
      3) parse Credit_History_Age (e.g., '10 Years and 9 Months') into:
         - Credit_History_Age_Year (float, e.g., 10.75)
         - Credit_History_Age_Month (int total months, e.g., 129)
    """
    try:
        # 1) Cast numeric-like strings (protect known categoricals/text)
        exclude = (
            "Customer_ID",
            "snapshot_date",
            "Type_of_Loan",
            "Credit_Mix",
            "Payment_Behaviour",
            "Credit_History_Age",
        )
        df = cast_to_numeric(df, exclude=exclude)

        # 2) Normalize Payment_Behaviour to 'Unknown' if not in whitelist / garbage
        valid_behaviours = [
            "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments",
        ]
        bad_tokens = ["na", "n/a", "none", "null", "-", "?", "unknown", "undefined", "nan"]

        raw_beh = F.trim(F.col("Payment_Behaviour"))
        only_letters_underscores = F.regexp_replace(raw_beh, r"[^A-Za-z_]", "")
        df = df.withColumn(
            "Payment_Behaviour",
            F.when(
                raw_beh.isNull()
                | (F.length(raw_beh) == 0)
                | F.lower(raw_beh).isin(bad_tokens)
                | (raw_beh != only_letters_underscores)
                | (~raw_beh.isin(valid_behaviours)),
                F.lit("Unknown"),
            ).otherwise(raw_beh)
        )

        # 3) Parse Credit_History_Age -> years/months
        #    Robust to singular/plural: "Year"/"Years", "Month"/"Months"
        #    Example input: "10 Years and 9 Months"
        cha_raw = F.col("Credit_History_Age")

        years_str  = F.regexp_extract(cha_raw, r"(?i)(\d+)\s*year", 1)
        months_str = F.regexp_extract(cha_raw, r"(?i)(\d+)\s*month", 1)

        df = df.withColumn(
            "__cha_years",
            F.when(F.length(years_str) == 0, None).otherwise(years_str.cast(IntegerType()))
        ).withColumn(
            "__cha_months",
            F.when(F.length(months_str) == 0, None).otherwise(months_str.cast(IntegerType()))
        )

        # compute outputs; if both parts missing -> null, else coalesce missing part to 0
        has_any = F.col("__cha_years").isNotNull() | F.col("__cha_months").isNotNull()
        yrs = F.coalesce(F.col("__cha_years"), F.lit(0))
        mos = F.coalesce(F.col("__cha_months"), F.lit(0))

        df = df.withColumn(
            "Credit_History_Age_Year",
            F.when(
                has_any,
                yrs.cast(FloatType()) + (mos.cast(FloatType()) / F.lit(12.0))
            ).otherwise(F.lit(None).cast(FloatType()))
        ).withColumn(
            "Credit_History_Age_Month",
            F.when(
                has_any,
                (yrs * F.lit(12) + mos)
            ).otherwise(F.lit(None).cast(IntegerType()))
        ).drop("__cha_years", "__cha_months")

        logger.info("Financials transformations: numeric sweep + Payment_Behaviour normalized + Credit_History_Age parsed.")
        return df

    except Exception as e:
        logger.error(f"Error in financials transformation: {str(e)}")
        raise
        

def transform_loan(df):
    """
    Loan: enforce id/date types, sweep numeric-like strings, then derive MOB/DPD.
    """
    try:
        # 1) Enforce key id/date types explicitly
        type_map = {
            "loan_id": StringType(),
            "Customer_ID": StringType(),
            "loan_start_date": DateType(),
            "snapshot_date": DateType(),
        }
        for col_name, dtype in type_map.items():
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast(dtype))

        # 2) Sweep numeric-like strings for the rest (exclude id/date)
        df = cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date", "loan_id", "loan_start_date"))

        # 3) Derived features (MOB, installments_missed, first_missed_date, DPD)
        df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

        # Avoid divide-by-zero for due_amt
        safe_due = when((col("due_amt").isNotNull()) & (col("due_amt") != 0), col("due_amt"))
        inst_missed = F.ceil(col("overdue_amt") / safe_due)
        df = df.withColumn(
            "installments_missed",
            when(inst_missed.isNotNull(), inst_missed).otherwise(0).cast(IntegerType())
        )

        df = df.withColumn(
            "first_missed_date",
            when(col("installments_missed") > 0,
                 F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
            ).cast(DateType())
        )

        df = df.withColumn(
            "dpd",
            when(col("overdue_amt") > 0.0,
                 F.datediff(col("snapshot_date"), col("first_missed_date"))
            ).otherwise(0).cast(IntegerType())
        )

        logger.info("Loan transformations: cast_to_numeric applied; MOB/DPD computed.")
        return df
    except Exception as e:
        logger.error(f"Error in loan transformation: {str(e)}")
        raise

# =============================================================================
# PROCESSORS
# =============================================================================
def process_silver_loan_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Load bronze loan CSV, apply transform_loan, save to silver parquet.
    """
    try:
        table_name = "lms_loan_daily"
        silver_dir = os.path.join(silver_directory, "loan_daily/")
        if not os.path.exists(silver_dir):
            os.makedirs(silver_dir)
            logger.info(f"Created directory: {silver_dir}")

        bronze_dir = os.path.join(bronze_directory, "lms_loan_daily/")
        partition_name = f"bronze_lms_loan_daily_{snapshot_date_str.replace('-','_')}.csv"
        filepath = os.path.join(bronze_dir, partition_name)

        if not os.path.exists(filepath):
            logger.warning(f"Bronze file not found: {filepath}")
            return None

        df = spark.read.csv(filepath, header=True, inferSchema=True)
        logger.info(f"Loaded {table_name} from {filepath}: {df.count()} rows")

        # Keep light sanity check only
        validate_customer_ids(df, table_name)

        df = transform_loan(df)

        out_file = os.path.join(silver_dir, f"silver_loan_daily_{snapshot_date_str.replace('-','_')}.parquet")
        df.write.mode("overwrite").parquet(out_file)
        logger.info(f"Saved loan_daily to {out_file}")
        return df
    except Exception as e:
        logger.error(f"Error processing loan table: {str(e)}")
        raise


def process_silver_clickstream_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Bronze → Silver for clickstream.
    """
    try:
        table_name = "feature_clickstream"
        logger.info(f"Starting clickstream processing for {snapshot_date_str}")

        df = load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark)
        if df is None:
            return None

        df = transform_clickstream(df)
        save_silver_table(df, snapshot_date_str, silver_directory, table_name)

        logger.info(f"Completed clickstream processing for {snapshot_date_str}")
        return df
    except Exception as e:
        logger.error(f"Error processing clickstream table: {str(e)}")
        raise


def process_silver_attributes_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Bronze → Silver for attributes.
    """
    try:
        table_name = "features_attributes"
        logger.info(f"Starting attributes processing for {snapshot_date_str}")

        df = load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark)
        if df is None:
            return None

        df = transform_attributes(df)
        save_silver_table(df, snapshot_date_str, silver_directory, table_name)

        logger.info(f"Completed attributes processing for {snapshot_date_str}")
        return df
    except Exception as e:
        logger.error(f"Error processing attributes table: {str(e)}")
        raise


def process_silver_financials_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Bronze → Silver for financials.
    """
    try:
        table_name = "features_financials"
        logger.info(f"Starting financials processing for {snapshot_date_str}")

        df = load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark)
        if df is None:
            return None

        df = transform_financials(df)
        save_silver_table(df, snapshot_date_str, silver_directory, table_name)

        logger.info(f"Completed financials processing for {snapshot_date_str}")
        return df
    except Exception as e:
        logger.error(f"Error processing financials table: {str(e)}")
        raise

# =============================================================================
# ORCHESTRATION
# =============================================================================
def get_table_processor(table_name):
    """
    Map table names to their processing functions.
    """
    table_function_mapping = {
        "lms_loan_daily": process_silver_loan_table,
        "feature_clickstream": process_silver_clickstream_table,
        "features_attributes": process_silver_attributes_table,
        "features_financials": process_silver_financials_table,
    }
    processor = table_function_mapping.get(table_name)
    if processor is None:
        error_msg = f"Unknown table name: {table_name}. Valid options: {list(table_function_mapping.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return processor


def process_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark, table_name=None):
    """
    Process one or all tables for a given snapshot date.
    """
    try:
        if table_name:
            logger.info(f"Processing single table: {table_name} for {snapshot_date_str}")
            processor = get_table_processor(table_name)
            result = processor(snapshot_date_str, bronze_directory, silver_directory, spark)
            logger.info(f"Successfully processed {table_name}")
            return result
        else:
            all_tables = ["lms_loan_daily", "feature_clickstream", "features_attributes", "features_financials"]
            results = {}

            logger.info("=" * 60)
            logger.info(f"Processing Silver Tables for {snapshot_date_str}")
            logger.info("=" * 60)

            for table in all_tables:
                logger.info(f"Processing {table}...")
                try:
                    processor = get_table_processor(table)
                    df = processor(snapshot_date_str, bronze_directory, silver_directory, spark)
                    results[table] = df
                    logger.info(f"✓ {table} completed")
                except Exception as e:
                    logger.error(f"✗ Error processing {table}: {str(e)}")
                    results[table] = None  # continue with others

            logger.info("=" * 60)
            logger.info(f"Silver layer processing completed for {snapshot_date_str}")
            logger.info("=" * 60)
            return results
    except Exception as e:
        logger.error(f"Critical error in process_silver_table: {str(e)}")
        raise
