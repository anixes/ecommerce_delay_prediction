from pathlib import Path

from loguru import logger
import pandas as pd
from sqlalchemy import create_engine, text
import typer

from delivery_delay_prediction.config import DATABASE_URL, INTERIM_DATA_DIR

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
        
        # Refresh the view definition from SQL file
        sql_path = Path("sql/feature_queries.sql")
        if sql_path.exists():
            logger.info("Refreshing analytical_dataset view definition...")
            with engine.connect() as conn:
                with open(sql_path, 'r', encoding='utf-8') as f:
                    # Split by semicolon and execute each part
                    for snippet in f.read().split(';'):
                        if snippet.strip():
                            conn.execute(text(snippet))
        
        query = "SELECT * FROM analytical_dataset"
        logger.info("Fetching data from analytical_dataset view...")
        df = pd.read_sql(query, engine)
        
        # Basic cleaning: Ensure timestamps are datetime objects
        datetime_cols = ['order_purchase_timestamp']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        logger.info(f"Successfully loaded {len(df)} rows. Columns: {df.columns.tolist()}")
        
        # Save to interim data directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"Dataset saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

if __name__ == "__main__":
    app()
