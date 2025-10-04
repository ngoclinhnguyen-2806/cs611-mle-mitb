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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("Processing dates:", dates_str_lst)

# =============================================================================
# BRONZE LAYER - Process all source tables
# =============================================================================
print("\n" + "="*80)
print("BRONZE LAYER PROCESSING")
print("="*80)

bronze_directory = "/apps/datamart/bronze/"

if not os.path.exists(bronze_directory):
    os.makedirs(bronze_directory)

# run bronze backfill for all tables
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_all_bronze_tables(
        date_str, 
        bronze_directory, 
        spark
    )

print("\n✓ Bronze layer backfill completed for all tables\n")

# =============================================================================
# SILVER LAYER - Process loan data
# =============================================================================
print("\n" + "="*80)
print("SILVER LAYER PROCESSING")
print("="*80)

silver_loan_daily_directory = "/app/datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
bronze_lms_directory = os.path.join(bronze_directory, "lms_loan_daily/")
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(
        date_str, 
        bronze_lms_directory, 
        silver_loan_daily_directory, 
        spark
    )

print("\n✓ Silver layer backfill completed\n")

# =============================================================================
# GOLD LAYER - Create label store
# =============================================================================
print("\n" + "="*80)
print("GOLD LAYER PROCESSING - LABEL STORE")
print("="*80)

gold_label_store_directory = "/app/datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(
        date_str, 
        silver_loan_daily_directory, 
        gold_label_store_directory, 
        spark, 
        dpd=30,  # Days Past Due threshold
        mob=6    # Months on Book threshold
    )

print("\n✓ Gold layer label store backfill completed\n")

# =============================================================================
# VERIFY RESULTS
# =============================================================================
print("\n" + "="*80)
print("VERIFYING RESULTS")
print("="*80)

# Check Bronze layer
print("\nBronze Layer Tables:")
for table_name in ['lms_loan_daily', 'features_clickstream', 'features_attributes', 'features_financials']:
    table_path = os.path.join(bronze_directory, table_name)
    if os.path.exists(table_path):
        file_count = len(glob.glob(os.path.join(table_path, '*.csv')))
        print(f"  {table_name}: {file_count} partitions")
    else:
        print(f"  {table_name}: NOT FOUND")

# Check Silver layer
print("\nSilver Layer:")
if os.path.exists(silver_loan_daily_directory):
    file_count = len(glob.glob(os.path.join(silver_loan_daily_directory, '*.csv')))
    print(f"  loan_daily: {file_count} partitions")

# Check Gold layer - Label Store
print("\nGold Layer - Label Store:")
folder_path = gold_label_store_directory
files_list = [os.path.join(folder_path, os.path.basename(f)) 
              for f in glob.glob(os.path.join(folder_path, '*'))]

if files_list:
    df = spark.read.option("header", "true").parquet(*files_list)
    print(f"  Total row count: {df.count()}")
    print(f"  Schema:")
    df.printSchema()
    print(f"\n  Sample data:")
    df.show(10)
else:
    print("  No data found in label store")

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETED")
print("="*80)