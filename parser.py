import pandas as pd

def process_and_save_excel(input_file_path, output_file_path):
    # Attempt to read the Excel file with openpyxl engine
    try:
        df = pd.read_excel(input_file_path, engine='openpyxl')
    except ValueError as e:
        print(f"Error reading the Excel file: {e}")
        return

    # Check if "Main Numbers" column exists
    if "Main Numbers" in df.columns:
        # Split the "Main Numbers" column into separate columns
        main_numbers_df = df["Main Numbers"].str.split(',', expand=True)
        
        # Rename columns to Number 1, Number 2, etc.
        column_names = [f"Number {i+1}" for i in range(main_numbers_df.shape[1])]
        main_numbers_df.columns = column_names
        
        # Save the processed data to a new Excel file
        main_numbers_df.to_excel(output_file_path, index=False, engine='openpyxl')
        print(f"Processed data has been saved to {output_file_path}")
    else:
        print("Column 'Main Numbers' not found in the input file.")

# Example usage
input_file_path = './correct_lotto_data.xlsx'  # Update this to your input .xlsx file path
output_file_path = 'parsed_lotto_data.xlsx'  # Update this to your desired output .xlsx file path
process_and_save_excel(input_file_path, output_file_path)
