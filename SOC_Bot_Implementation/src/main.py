import click
import pandas as pd

from src.logger import get_logger
from src.preprocessing.preprocess import preprocess_bulk_alerts
from src import config
# from src.models.model_training import train_false_positive_detector

@click.command()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose (DEBUG) logging.")
@click.option("--excel-path", default=config.DEFAULT_EXCEL_PATH, help="Path to the Excel file with alerts.")
def main(verbose: bool, excel_path: str):

    config.VERBOSE = verbose

    logger = get_logger(__name__)

    logger.info("Starting SOC Bot Preprocessing...")
    logger.debug(f"Excel path received: {excel_path}")

    # Load data
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        return

    logger.info(f"Data loaded successfully. Shape: {df.shape}")

    # Preprocess
    try:
        df_clean = preprocess_bulk_alerts(df)
    except Exception as e:
        logger.error(f"Error in bulk preprocessing: {e}")
        return

    logger.info("Data preprocessing completed.")
    logger.debug(f"Sample post-preprocessing data:\n{df_clean.head()}")

    # TODO: Possibly call your training logic here
    # model = train_false_positive_detector(df_clean)

    logger.info("Done.")

if __name__ == "__main__":
    main()
