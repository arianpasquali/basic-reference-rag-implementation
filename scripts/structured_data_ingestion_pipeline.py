#!/usr/bin/env python3
"""
Simple script to ingest Toyota sales data into SQLite database.
Hard-coded CSV file paths for simplicity.
"""

import logging
from pathlib import Path
import sqlite3

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_csv_to_sqlite():
    """Ingest all CSV files into SQLite database."""

    # Database configuration
    db_path = "toyota_sales.db"
    data_dir = Path("data")

    # CSV files and their table names (hard-coded)
    csv_files = {
        "DIM_COUNTRY.csv": "dim_country",
        "DIM_MODEL.csv": "dim_model",
        "DIM_ORDERTYPE.csv": "dim_ordertype",
        "FACT_SALES.csv": "fact_sales",
        "FACT_SALES_ORDERTYPE.csv": "fact_sales_ordertype",
    }

    # Connect to SQLite database (creates if not exists)
    logger.info(f"Connecting to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        total_rows = 0

        # Process each CSV file
        for csv_file, table_name in csv_files.items():
            file_path = data_dir / csv_file

            logger.info(f"Processing {csv_file}...")

            # Load CSV into DataFrame
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {csv_file}")

            # Write data to SQLite table (replace if exists)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"Created table '{table_name}' with {len(df)} rows")

            total_rows += len(df)

        logger.info(f"Successfully ingested {total_rows} total rows into {len(csv_files)} tables")

        # Show table information
        logger.info("\nDatabase summary:")
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            logger.info(f"  {table_name}: {count:,} rows")

        # Run example queries
        logger.info("\nRunning example queries...")

        # Query 1: Simple country list
        logger.info("\nQuery 1: All countries")
        cursor.execute("SELECT * FROM dim_country")
        countries = cursor.fetchall()
        for row in countries:
            print(f"  {row}")

        # Query 2: Total sales by country
        logger.info("\nQuery 2: Total sales by country")
        query = """
        SELECT
            c.country,
            c.region,
            SUM(f.contracts) as total_contracts
        FROM fact_sales f
        JOIN dim_country c ON f.country_code = c.country_code
        GROUP BY c.country, c.region
        ORDER BY total_contracts DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            print(f"  {row[0]:<15} {row[1]:<15} {row[2]:>8}")

        # Query 3: Sales by brand
        logger.info("\nQuery 3: Sales by brand and powertrain")
        query = """
        SELECT
            m.brand,
            m.powertrain,
            SUM(f.contracts) as total_contracts
        FROM fact_sales f
        JOIN dim_model m ON f.model_id = m.model_id
        GROUP BY m.brand, m.powertrain
        ORDER BY total_contracts DESC
        """
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            print(f"  {row[0]:<8} {row[1]:<8} {row[2]:>8}")

        logger.info("\nData ingestion completed successfully!")
        logger.info(f"Database file: {db_path}")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise
    finally:
        # Close connection
        conn.close()
        logger.info("Database connection closed")


def main():
    """Main function."""
    logger.info("Starting Toyota sales data ingestion to SQLite...")
    ingest_csv_to_sqlite()


if __name__ == "__main__":
    main()
