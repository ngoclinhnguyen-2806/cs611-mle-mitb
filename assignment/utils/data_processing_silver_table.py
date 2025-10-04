import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_loan_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Process loan data from bronze to silver layer with transformations
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        bronze_directory: Bronze layer base directory
        silver_directory: Silver layer directory for loan data
        spark: SparkSession object
    
    Returns:
        DataFrame: Processed Spark DataFrame
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    bronze_lms_directory = os.path.join(bronze_directory, "lms_loan_daily/")
    partition_name = "bronze_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_lms_directory, partition_name)
    
    if not os.path.exists(filepath):
        print(f"Warning: Bronze file not found: {filepath}")
        return None
    
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", 
                       F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", 
                       F.when(col("installments_missed") > 0, 
                              F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", 
                       F.when(col("overdue_amt") > 0.0, 
                              F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df


def process_silver_feature_table(snapshot_date_str, bronze_directory, silver_directory, spark, table_name):
    """
    Process feature tables from bronze to silver layer
    Currently just creates directory structure - processing logic to be added later
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        bronze_directory: Bronze layer base directory
        silver_directory: Silver layer base directory
        spark: SparkSession object
        table_name: Name of the feature table
    
    Returns:
        None
    """
    # Create table-specific directory in silver layer
    table_silver_directory = os.path.join(silver_directory, table_name + "/")
    
    if not os.path.exists(table_silver_directory):
        os.makedirs(table_silver_directory)
        print(f"Created directory: {table_silver_directory}")
    
    print(f"Silver layer directory ready for {table_name}")
    print(f"Note: Processing logic for {table_name} to be implemented in future iterations")
    
    return None


def process_all_silver_tables(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Process all silver tables for a given snapshot date
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        bronze_directory: Bronze layer base directory
        silver_directory: Silver layer base directory
        spark: SparkSession object
    
    Returns:
        dict: Dictionary of table_name -> DataFrame (or None for feature tables)
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Processing Silver Tables for {snapshot_date_str}")
    print(f"{'='*60}\n")
    
    # Process loan data with full transformations
    print("Processing lms_loan_daily (with transformations)...")
    try:
        silver_loan_directory = os.path.join(silver_directory, "loan_daily/")
        if not os.path.exists(silver_loan_directory):
            os.makedirs(silver_loan_directory)
        
        df = process_silver_loan_table(
            snapshot_date_str, 
            bronze_directory, 
            silver_loan_directory, 
            spark
        )
        results['lms_loan_daily'] = df
        print(f"✓ lms_loan_daily completed\n")
    except Exception as e:
        print(f"✗ Error processing lms_loan_daily: {e}\n")
        results['lms_loan_daily'] = None
    
    # Process feature tables (directory creation only for now)
    feature_tables = [
        'features_clickstream',
        'features_attributes', 
        'features_financials'
    ]
    
    for table_name in feature_tables:
        print(f"Processing {table_name} (directory setup only)...")
        try:
            process_silver_feature_table(
                snapshot_date_str,
                bronze_directory,
                silver_directory,
                spark,
                table_name
            )
            results[table_name] = None
            print(f"✓ {table_name} directory created\n")
        except Exception as e:
            print(f"✗ Error processing {table_name}: {e}\n")
            results[table_name] = None
    
    print(f"{'='*60}")
    print(f"Silver layer processing completed for {snapshot_date_str}")
    print(f"{'='*60}\n")
    
    return results


# Backward compatibility: keep the original function signature
def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    """
    Original function maintained for backward compatibility
    Processes loan data only
    """
    # Extract bronze base directory from lms directory path
    bronze_directory = bronze_lms_directory.replace("lms_loan_daily/", "").rstrip('/')
    
    return process_silver_loan_table(
        snapshot_date_str,
        bronze_directory,
        silver_loan_daily_directory,
        spark
    )