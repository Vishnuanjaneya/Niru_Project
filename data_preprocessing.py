import pandas as pd


def load_data(file_path):
    """Load the dataset from the specified CSV file."""
    df = pd.read_csv(file_path, sep="\t", engine='python')
    return df


def encode_categorical_variables(df):
    """Convert categorical variables into numerical format using one-hot encoding."""
    df = pd.get_dummies(df, columns=['qualification', 'skills'], drop_first=True)
    return df


def preprocess_data(file_path):
    """Preprocess the dataset."""
    df = load_data(file_path)

    # Drop the 'name' column as it is not needed for model training
    df.drop(columns=['name'], inplace=True)

    # Fill missing values for numerical columns
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing values for categorical columns
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna('Unknown')

    # Convert categorical variables into numerical format
    df = encode_categorical_variables(df)

    # Ensure all numeric data types are floats
    for column in df.select_dtypes(include=['int64', 'bool']).columns:
        df[column] = df[column].astype(float)  # Convert int and bool columns to float

    # Save the preprocessed DataFrame to a CSV file
    output_path = r"C:\Users\ADMIN\pythonProject7\data\preprocessed_candidates.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

    return df


if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\ADMIN\pythonProject7\data\candidates.csv"  # Update with your path
    df = preprocess_data(DATASET_PATH)
    print("Preprocessed DataFrame:")
    print(df.head())
