from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from delivery_delay_prediction.config import INTERIM_DATA_DIR, DATABASE_URL

import pandas as pd
from sqlalchemy import create_engine
import typer
from loguru import logger
from delivery_delay_prediction.config import DATABASE_URL, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    output_path: Path = INTERIM_DATA_DIR / "analytical_dataset.csv",
):
    """
    Fetches the pre-calculated analytical view from MySQL and saves it locally.
    """
    logger.info("Connecting to MySQL database...")
    try:
        engine = create_engine(DATABASE_URL)
        query = "SELECT * FROM analytical_dataset"
        
        logger.info("Fetching data from analytical_dataset view (this may take a few seconds)...")
        df = pd.read_sql(query, engine)
        
        # Basic cleaning: Ensure timestamps are datetime objects
        datetime_cols = ['order_purchase_timestamp']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        logger.info(f"Successfully loaded {len(df)} rows.")
        
        # Save to processed data directory for caching/offline use
        df.to_csv(output_path, index=False)
        logger.success(f"Dataset saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

if __name__ == "__main__":
    app()
