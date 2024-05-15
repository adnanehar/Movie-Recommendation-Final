import pandas as pd

def test_data(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        print(missing_values)
    else:
        print("No missing values found.")

    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows:
        print("Duplicate rows found:", duplicate_rows)
    else:
        print("No duplicate rows found.")

    # Check data types
    print("Data types:")
    print(df.dtypes)

    # Check for unique values in categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        unique_values = df[column].unique()
        print(f"Unique values in {column}: {unique_values}")

    # Check for outliers in numerical columns
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    for column in numerical_columns:
        # Use 1.5*IQR rule to detect outliers
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not outliers.empty:
            print(f"Outliers found in {column}:")
            print(outliers)

    # Check for correlations between numerical columns
    correlation_matrix = df.corr()
    print("Correlation matrix:")
    print(correlation_matrix)

    # Add more tests as needed

# Example usage
test_data(r'C:\Users\dell\Documents\movies_metadata.csv')

