import pandas as pd
import os

def count_videos_by_year(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Read the CSV file
        # Using on_bad_lines='skip' or error_bad_lines=False (depending on version) to handle potential malformed lines
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
        except TypeError:
            # Fallback for older pandas versions
            df = pd.read_csv(file_path, error_bad_lines=False)
        
        # Check if 'publishedAt' column exists
        if 'publishedAt' not in df.columns:
            print("Error: 'publishedAt' column not found in the CSV.")
            return

        # Convert 'publishedAt' to datetime objects
        # coerce errors will turn unparseable dates into NaT
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')

        # Drop rows where publishedAt is NaT (if any)
        df = df.dropna(subset=['publishedAt'])

        # Extract year
        df['year'] = df['publishedAt'].dt.year

        # Count occurrences of each year
        year_counts = df['year'].value_counts().sort_index()

        # Print the results
        print("Year Counts:")
        print(year_counts)
        
        # Optionally save to a file
        # year_counts.to_csv('year_counts.csv')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Adjust path if necessary. Based on previous LS, it seems to be in code/Youtube_Videos.csv
    # But let's check relative to current script location.
    # Assuming script is in .../code/ and csv is in .../code/code/
    
    file_path = os.path.join('code', 'Youtube_Videos.csv')
    
    # Verify if the path is correct relative to CWD, otherwise try absolute path or direct filename
    if not os.path.exists(file_path):
        # Try just the filename if it's in the same directory
        if os.path.exists('Youtube_Videos.csv'):
            file_path = 'Youtube_Videos.csv'
        else:
             # Try absolute path based on previous context
             file_path = r'C:\Users\uu\Desktop\大三下\统计分析与建模\final project\code\code\Youtube_Videos.csv'

    print(f"Reading file from: {file_path}")
    count_videos_by_year(file_path)
