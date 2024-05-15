from matplotlib import _preprocess_data
import pandas as pd
import pytest
import logging
from unittest.mock import MagicMock, patch

from steps.clean_data import clean_data



sample_csv_path = r'C:\Users\dell\Documents\AJAX-Movie-Recommendation\cleaned_data.csv'  

@pytest.fixture
def test_successful_data_load():
    # Arrange
    test_csv_path = r'C:\Users\dell\Documents\AJAX-Movie-Recommendation\cleaned_data.csv'

    # Act
    result = clean_data(test_csv_path)

    # Assert
    assert isinstance(result, pd.DataFrame)

def test_file_not_found():
    # Arrange
    non_existent_file_path = r'C:\Users\dell\Documents\AJAX-Movie-Recommendation\cleaned_data.csv'

    # Act and Assert
    with pytest.raises(FileNotFoundError):
        clean_data(non_existent_file_path)

def log_capture():
    class LogCaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = LogCaptureHandler()
    logging.getLogger().addHandler(handler)
    yield handler.records
    logging.getLogger().removeHandler(handler)

def test_file_not_found(log_capture):
    # Arrange
    non_existent_file_path = r'C:\Users\dell\Documents\AJAX-Movie-Recommendation\cleaned_data.csv'

    # Act
    with pytest.raises(FileNotFoundError):
        clean_data(non_existent_file_path)

    # Assert
    assert len(log_capture) > 0
    assert "File not found" in log_capture[0].message

def test_parsing_error(log_capture):
    # Arrange
    invalid_csv_path = r'C:\Users\dell\Documents\AJAX-Movie-Recommendation\cleaned_data.csv'

    # Act
    with pytest.raises(pd.errors.ParserError):
        clean_data(invalid_csv_path)

    # Assert
    assert len(log_capture) > 0
    assert "Error parsing the file" in log_capture[0].message

def test_null_values_in_data(log_capture):
    # Arrange
    csv_with_nulls_path = r'C:\Users\dell\Documents\AJAX-Movie-Recommendation\cleaned_data.csv'

    # Act
    result = clean_data(csv_with_nulls_path)

    # Assert
    assert "Null values found in the dataset" in [record.message for record in log_capture]
    # Additional assertions for the result

def test_missing_columns():
    # Arrange
    data = pd.DataFrame({'movie_id': [1, 2, 3], 'title': ['Movie 1', 'Movie 2', 'Movie 3']})  # Only 'movie_id' and 'title' columns

    # Act and Assert
    with pytest.raises(ValueError):
        _preprocess_data(data)

def test_complete_preprocessing():
    # Arrange
    data = pd.DataFrame({'movie_id': [1, 2, 3], 'title': ['Movie 1', 'Movie 2', 'Movie 3'], 'genre': ['Action', 'Comedy', 'Drama']})

    # Act
    processed_data = _preprocess_data(data)

    # Assert (Add appropriate assertions as needed)

def test_missing_values_visualization(capsys):
    # Arrange
    data_with_missing_values = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, 6]})
    data_without_missing_values = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Act
    visualize_data(data_with_missing_values)
    captured = capsys.readouterr()

    # Assert
    assert "No missing values found in the dataset." not in captured.out

    # Repeat for data_without_missing_values
    visualize_data(data_without_missing_values)
    captured = capsys.readouterr()
    assert "No missing values found in the dataset." in captured.out

# Define other tests similarly...
