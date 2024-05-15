import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input data.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        try:
            # Fill missing values, convert data types, etc. as needed
             # data["imdb_score"] = data["imdb_score"].fillna(data["imdb_score"].mode()[0])
            #  data["aspect_ratio"] = data["aspect_ratio"].fillna(data["aspect_ratio"].mode()[0])
             # data["movie_facebook_likes"] = data["movie_facebook_likes"].fillna(data["movie_facebook_likes"].mode()[0])
             # data["actor_2_facebook_likes"] = data["actor_2_facebook_likes"].fillna(data["actor_2_facebook_likes"].mode()[0])
            data["title_year"] = data["title_year"].fillna(data["title_year"].mode()[0])

            data = data.drop(['title_year', 'duration'], axis=1)


            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("duration", axis=1)
            y = data["duration"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)

def main():
    # Load your dataset
    data = pd.read_csv(r'C:\Users\dell\Documents\AJAX-Movie-Recommendation\cleaned_data.csv')  # Replace 'your_dataset.csv' with your actual dataset file path
    
    # Initialize data preprocessing strategy
    preprocess_strategy = DataPreprocessStrategy()
    
    # Initialize data cleaning instance
    cleaning = DataCleaning(data, preprocess_strategy)
    
    # Perform data cleaning and preprocessing
    cleaned_data = cleaning.handle_data()
    
    # Display the cleaned data
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    
    # Initialize data divide strategy
    divide_strategy = DataDivideStrategy()
    
    # Divide the cleaned data into train and test sets
    X_train, X_test, y_train, y_test = divide_strategy.handle_data(cleaned_data)
    
    # Display the shapes of train and test sets
    print("Train set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

if __name__ == "__main__":
    main()
