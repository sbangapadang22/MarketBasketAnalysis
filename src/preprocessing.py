import pandas as pd
import os

def preprocess_data(file_path):
    try:
        # Load the dataset with comma-separated values
        data = pd.read_csv(file_path, sep=',')

        # Print the first few rows to verify data loading
        print("First few rows of the dataset:")
        print(data.head())

        # Print the column names to ensure they are read correctly
        print("Column names in the dataset:")
        print(data.columns)

        # Convert InvoiceDate to datetime format with specified format
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%y %H:%M')

        # Ensure numerical values are in the correct type
        data['Quantity'] = data['Quantity'].astype(int)
        data['UnitPrice'] = data['UnitPrice'].astype(float)

        # Grouping the data by InvoiceNo and Description and summing up the Quantity
        basket = data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')

        # Convert quantities to binary (presence/absence of items)
        basket = basket.map(lambda x: 1 if x > 0 else 0)

        return basket
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '../data/OnlineRetail.csv')
    output_path = os.path.join(current_dir, '../data/Processed_OnlineRetail.csv')

    # Preprocess the data
    processed_data = preprocess_data(file_path)

    if processed_data is not None:
        # Save the preprocessed data to a new CSV file
        processed_data.to_csv(output_path)
        print("Data preprocessing complete. The preprocessed data is saved as 'Processed_OnlineRetail.csv'.")
    else:
        print("Data preprocessing failed.")
