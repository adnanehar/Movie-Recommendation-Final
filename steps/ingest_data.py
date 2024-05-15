import logging
import pandas as pd
from zenml import step

class IngestData:

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        # Read the CSV file
        df = pd.read_csv(self.data_path)

        # Convert columns to appropriate data types
        
        df['title_year'] = df['title_year'].astype('string')  # Assuming 'director_name' is a string
    


        return df

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Args:
        data_path: str: Path to the data file.
    Returns:
        df: pd.DataFrame: DataFrame containing the ingested data.
    """
    try:
        ingest = IngestData(data_path)
        df = ingest.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e
