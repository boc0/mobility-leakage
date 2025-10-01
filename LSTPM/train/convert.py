import pandas as pd
import argparse
import json
import os
from datetime import datetime, timedelta

def convert_csv_to_deepmove_format(csv_path, out_txt_path, create_metadata=False):
    """
    Convert a single CSV into DeepMove format, optionally generating a metadata.json file.
    """
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Create pid mapping from lat/lon
    df_locations = df[['lat','lon']].drop_duplicates().reset_index(drop=True)
    df_locations['pid'] = df_locations.index.map(str)
    
    # Merge pid into the main dataframe
    df = df.merge(df_locations, on=['lat','lon'], how='left')

    # Use existing timestamp if present; otherwise convert from day/hour
    if 'timestamp' not in df.columns:
        start_date = datetime(2025, 6, 30)
        df['timestamp'] = df.apply(
            lambda row: (start_date + timedelta(days=row['day'], hours=row['hour'])).strftime('%Y-%m-%d %H:%M:%S'),
            axis=1
        )
    
    # Prepare the result dataframe in DeepMove format
    result = pd.DataFrame()
    result['tid'] = df['tid']
    result['lat'] = df['lat']
    result['lon'] = df['lon']
    result['timestamp'] = df['timestamp']
    result['pid'] = df['pid']
    
    # Ensure output directory exists and save the file
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    result.to_csv(out_txt_path, sep='\u0001', header=False, index=False, encoding='utf-8')
    print(f"Saved {len(df)} records to {out_txt_path}.")

    # Optionally, create and save the metadata.json file
    if create_metadata:
        pid_mapping = {row.pid: [row.lat, row.lon] for row in df_locations.itertuples()}
        users = df['tid'].unique().tolist()
        metadata = {
            "pid_mapping": pid_mapping,
            "users": users,
        }
        metadata_file = os.path.join(os.path.dirname(out_txt_path), 'metadata.json')
        with open(metadata_file, 'w') as jf:
            json.dump(metadata, jf, indent=2)
        print(f"Saved metadata to {metadata_file}")
        print(f"Found {len(pid_mapping)} locations and {len(users)} users.")

def convert_directory(in_dir, out_dir='preprocessed'):
    """
    Converts all CSV files in a directory, creates preprocessed text files in
    an output directory, and generates a single, unified metadata.json.
    """
    print(f"Processing all CSV files from directory: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(in_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the input directory.")
        return

    # 1. Read all CSVs to create a combined view for global mappings
    all_dfs = [pd.read_csv(os.path.join(in_dir, f)) for f in csv_files]
    '''
    for df in all_dfs:
        # round lat/lon to 6 decimal places to avoid floating point issues
        df['lat'] = df['lat'].round(6)
        df['lon'] = df['lon'].round(6)
    '''
    for file, df in zip(csv_files, all_dfs):
    if not all(col in df.columns for col in ['tid', 'lat', 'lon', 'timestamp']):
        raise ValueError(f"CSV file {file} must contain 'tid', 'lat', 'lon', and "
        f"'timestamp' columns, has instead: {df.columns.tolist()}")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 2. Create one unified pid mapping for all locations
    print("Generating unified location mapping...")
    df_locations = combined_df[['lat','lon']].drop_duplicates().reset_index(drop=True)
    df_locations['pid'] = df_locations.index.map(str)
    pid_mapping = {row.pid: [row.lat, row.lon] for row in df_locations.itertuples()}
    
    # 3. Create and save the unified metadata.json
    users = combined_df['tid'].unique().tolist()
    metadata = {
        "pid_mapping": pid_mapping,
        "users": users,
    }
    metadata_file = os.path.join(out_dir, 'metadata.json')
    with open(metadata_file, 'w') as jf:
        json.dump(metadata, jf, indent=2)
    print(f"Saved unified metadata for {len(pid_mapping)} locations and {len(users)} users to {metadata_file}")

    # 4. Process and save each file individually using the unified mapping
    for i, df in enumerate(all_dfs):
        csv_file_name = csv_files[i]
        print(f"Converting {csv_file_name}...")
        
        # Merge with the global location mapping to get pids
        df = df.merge(df_locations, on=['lat','lon'], how='left')

        # Handle timestamp
        if 'timestamp' not in df.columns:
            start_date = datetime(2025, 6, 30)
            df['timestamp'] = df.apply(
                lambda row: (start_date + timedelta(days=row['day'], hours=row['hour'])).strftime('%Y-%m-%d %H:%M:%S'),
                axis=1
            )
        
        # Prepare the result dataframe
        result = pd.DataFrame()
        result['tid'] = df['tid']
        result['lat'] = df['lat']
        result['lon'] = df['lon']
        result['timestamp'] = df['timestamp']
        result['pid'] = df['pid']
        
        # Save the converted file
        out_filename = os.path.splitext(csv_file_name)[0] + '.txt'
        out_txt_path = os.path.join(out_dir, out_filename)
        result.to_csv(out_txt_path, sep='\u0001', header=False, index=False, encoding='utf-8')
        print(f"Saved converted file to {out_txt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert trajectory CSV data to DeepMove format.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Single file conversion command
    parser_file = subparsers.add_parser('file', help='Convert a single CSV file.')
    parser_file.add_argument('--in_csv', type=str, help='Input CSV file path', required=True)
    parser_file.add_argument('--out_txt', type=str, help='Output TXT file path', required=True)
    parser_file.add_argument('--create_metadata', action='store_true', help='Flag to create metadata.json')

    # Directory conversion command
    parser_dir = subparsers.add_parser('dir', help='Convert all CSV files in a directory.')
    parser_dir.add_argument('--in_dir', type=str, help='Input directory path', required=True)
    parser_dir.add_argument('--out_dir', type=str, default='preprocessed', help='Output directory path')
    
    args = parser.parse_args()

    if args.command == 'file':
        convert_csv_to_deepmove_format(args.in_csv, args.out_txt, args.create_metadata)
    elif args.command == 'dir':
        convert_directory(args.in_dir, args.out_dir)
