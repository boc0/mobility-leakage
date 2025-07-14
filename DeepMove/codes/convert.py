import pandas as pd
from datetime import datetime, timedelta

def convert_csv_to_deepmove_format(input_csv_path, output_txt_path):
    """
    Converts a custom trajectory CSV to the format expected by DeepMove's
    sparse_traces.py script.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_txt_path (str): Path to save the converted text file.
    """
    print(f"Reading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    # 1. Create unique Venue IDs (pid) from lat/lon
    # We can create a string representation for each unique coordinate pair.
    df['pid'] = df.groupby(['lat', 'lon']).ngroup().astype(str)

    # 2. Create a valid timestamp from 'day' and 'hour'
    # The script expects a "YYYY-MM-DD HH:MM:SS" format.
    # We'll create synthetic dates, assuming 'day' is day of the week (0-6).
    # Let's use the first week of July 2025 as a base.
    start_date = datetime(2025, 6, 30) # A Monday
    df['timestamp'] = df.apply(
        lambda row: (start_date + timedelta(days=row['day'], hours=row['hour'])).strftime('%Y-%m-%d %H:%M:%S'),
        axis=1
    )

    # 3. Prepare the final DataFrame with all required columns
    # The expected format is:
    # Record ID, User ID, Lat, Lon, Timestamp, Offset, Venue Cat, Tweet, Venue ID
    # We will use placeholders for columns we don't have.
    result_df = pd.DataFrame()
    result_df['record_id'] = range(len(df))
    result_df['uid'] = df['label']
    result_df['lat'] = df['lat']
    result_df['lon'] = df['lon']
    result_df['timestamp'] = df['timestamp']
    result_df['offset'] = '0'  # Placeholder
    result_df['venue_cat'] = df['category'] # Using your category
    result_df['tweet'] = 'tweet' # Placeholder
    result_df['pid'] = df['pid']

    print(f"Saving converted data to {output_txt_path}...")
    # 4. Save to the ``-separated text file
    result_df.to_csv(
        output_txt_path,
        sep='',
        header=False,
        index=False,
        encoding='utf-8'
    )
    print("Conversion complete.")
    print(f"Generated {len(df)} records and {df['pid'].nunique()} unique venues.")


if __name__ == '__main__':
    # Replace with the actual path to your CSV file
    your_csv_file = 'small.csv'
    # This should be the path sparse_traces.py expects
    output_file = 'data/foursquare/tweets_clean.txt'

    # Make sure the output directory exists
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    convert_csv_to_deepmove_format(your_csv_file, output_file)
