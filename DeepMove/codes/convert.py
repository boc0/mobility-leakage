import pandas as pd
import argparse
import json
import os
from datetime import datetime, timedelta

def convert_csv_to_deepmove_format(train_csv_path, test_csv_path, train_out_txt, test_out_txt):
    """
    Convert train/test CSVs into DeepMove format, generate venues.json mapping.
    """
    print(f"Reading train data from {train_csv_path} and test data from {test_csv_path}...")
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # create unified pid mapping across both datasets
    df_all = pd.concat([df_train[['lat','lon']], df_test[['lat','lon']]]).drop_duplicates().reset_index(drop=True)
    df_all['pid'] = df_all.index.map(str)
    pid_mapping = {row.pid: [row.lat, row.lon] for row in df_all.itertuples()}

    # merge pid into train/test
    df_train = df_train.merge(df_all, on=['lat','lon'], how='left')
    df_test = df_test.merge(df_all, on=['lat','lon'], how='left')

    # helper to convert a df and output path
    def to_deepmove(df, out_path):
        # use existing timestamp if present; otherwise convert from day/hour
        if 'timestamp' not in df.columns:
            start_date = datetime(2025, 6, 30)
            df['timestamp'] = df.apply(
                lambda row: (start_date + timedelta(days=row['day'], hours=row['hour'])).strftime('%Y-%m-%d %H:%M:%S'),
                axis=1
            )
        result = pd.DataFrame()
        result['tid'] = df['tid']
        result['lat'] = df['lat']
        result['lon'] = df['lon']
        result['timestamp'] = df['timestamp']
        # result['venue_cat'] = df['category']
        result['pid'] = df['pid']
        result.to_csv(out_path, sep='\u0001', header=False, index=False, encoding='utf-8')
        print(f"Saved {len(df)} records to {out_path}.")
    

    def to_deepmove_test(df, out_path):
        # use existing timestamp if present; otherwise convert from day/hour
        if 'timestamp' not in df.columns:
            start_date = datetime(2025, 6, 30)
            df['timestamp'] = df.apply(
                lambda row: (start_date + timedelta(days=row['day'], hours=row['hour'])).strftime('%Y-%m-%d %H:%M:%S'),
                axis=1
            )
        result = pd.DataFrame()
        result['tid'] = df['tid']
        result['lat'] = df['lat']
        result['lon'] = df['lon']
        result['timestamp'] = df['timestamp']
        # result['venue_cat'] = df['category']
        result['pid'] = df['pid']
        result.to_csv(out_path, sep='\u0001', header=False, index=False, encoding='utf-8')
        print(f"Saved {len(df)} records to {out_path}.")


    # ensure output dirs exist
    os.makedirs(os.path.dirname(train_out_txt), exist_ok=True)
    os.makedirs(os.path.dirname(test_out_txt), exist_ok=True)
    to_deepmove(df_train, train_out_txt)
    to_deepmove_test(df_test, test_out_txt)

    # write metadata.json instead of venues.json
    metadata_file = os.path.join(os.path.dirname(train_out_txt), 'metadata.json')
    # instead we will save a list of the unique user IDs
    users = pd.concat([df_train['tid'], df_test['tid']]).unique().tolist()
    metadata = {
        "pid_mapping": pid_mapping,
        "users": users,
    }
    with open(metadata_file, 'w') as jf:
        json.dump(metadata, jf)
    print(f"Saved metadata to {metadata_file}")
    print(f"Found {len(pid_mapping)} locations and {len(users)} users.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, help='train CSV file', required=True)
    parser.add_argument('--secondary_csv', type=str, help='test CSV file', required=True)
    parser.add_argument('--train_txt', type=str, help='output train txt', required=True)
    parser.add_argument('--secondary_txt', type=str, help='output test txt', required=True)
    args = parser.parse_args()
    convert_csv_to_deepmove_format(args.train_csv, args.secondary_csv, args.train_txt, args.secondary_txt)
