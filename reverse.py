import pandas as pd

def reverse_excel_order(file_path, output_file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Keep the header row, reverse the order of the rest
    reversed_df = pd.concat([df.iloc[:1], df.iloc[1:].iloc[::-1]])

    # Save the reversed DataFrame to a new Excel file
    reversed_df.to_excel(output_file_path, index=False)

if __name__ == "__main__":
    file_path = './lotto_data.xlsx'  # Update this to your source Excel file path
    output_file_path = './correct_lotto_data.xlsx'  # Update this to your desired output file path
    reverse_excel_order(file_path, output_file_path)
