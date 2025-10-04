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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    """
    Process label store in gold layer from silver loan data
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        silver_loan_daily_directory: Silver layer directory for loan data
        gold_label_store_directory: Gold layer directory for label store
        spark: SparkSession object
        dpd: Days Past Due threshold for default definition
        mob: Months on Book threshold (minimum loan age)
    
    Returns:
        DataFrame: Processed Spark DataFrame with labels
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_loan_daily_directory, partition_name)
    
    if not os.path.exists(filepath):
        print(f"Warning: Silver file not found: {filepath}")
        return None
    
    df = spark.read.parquet(filepath)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(gold_label_store_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df


def process_feature_store_gold_table(snapshot_date_str, silver_directory, gold_feature_store_directory, spark):
    """
    Process feature store in gold layer from silver feature tables
    Currently just creates directory structure - processing logic to be added later
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        silver_directory: Silver layer base directory
        gold_feature_store_directory: Gold layer directory for feature store
        spark: SparkSession object
    
    Returns:
        None
    """
    # Create feature store directory
    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)
        print(f"Created directory: {gold_feature_store_directory}")
    
    print(f"Gold feature store directory ready")
    print(f"Note: Feature store processing logic to be implemented in future iterations")
    print(f"      Will combine: features_clickstream, features_attributes, features_financials")
    
    return None


def process_all_gold_tables(snapshot_date_str, silver_directory, gold_directory, spark, dpd=30, mob=6):
    """
    Process all gold tables for a given snapshot date
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        silver_directory: Silver layer base directory
        gold_directory: Gold layer base directory
        spark: SparkSession object
        dpd: Days Past Due threshold for default definition
        mob: Months on Book threshold (minimum loan age)
    
    Returns:
        dict: Dictionary of table_name -> DataFrame
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Processing Gold Tables for {snapshot_date_str}")
    print(f"{'='*60}\n")
    
    # Process label store
    print(f"Processing label store (dpd={dpd}, mob={mob})...")
    try:
        gold_label_store_directory = os.path.join(gold_directory, "label_store/")
        if not os.path.exists(gold_label_store_directory):
            os.makedirs(gold_label_store_directory)
        
        silver_loan_daily_directory = os.path.join(silver_directory, "loan_daily/")
        
        df = process_labels_gold_table(
            snapshot_date_str,
            silver_loan_daily_directory,
            gold_label_store_directory,
            spark,
            dpd,
            mob
        )
        results['label_store'] = df
        print(f"✓ label_store completed\n")
    except Exception as e:
        print(f"✗ Error processing label_store: {e}\n")
        results['label_store'] = None
    
    # Process feature store (directory setup only for now)
    print("Processing feature store (directory setup only)...")
    try:
        gold_feature_store_directory = os.path.join(gold_directory, "feature_store/")
        
        process_feature_store_gold_table(
            snapshot_date_str,
            silver_directory,
            gold_feature_store_directory,
            spark
        )
        results['feature_store'] = None
        print(f"✓ feature_store directory created\n")
    except Exception as e:
        print(f"✗ Error processing feature_store: {e}\n")
        results['feature_store'] = None
    
    print(f"{'='*60}")
    print(f"Gold layer processing completed for {snapshot_date_str}")
    print(f"{'='*60}\n")
    
    return results