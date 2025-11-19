import argparse
import os
import json

def load_metadata(meta_path):
    with open(meta_path, 'r') as f:
        return json.load(f)

def users_from_txt(txt_path, users_original):
    ids = set()
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                continue
            try:
                uid_int = int(parts[0])
            except ValueError:
                continue
            if 0 <= uid_int < len(users_original):
                ids.add(users_original[uid_int])
    return ids

def users_from_csv(csv_path):
    ids = set()
    with open(csv_path, 'r') as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split(',')
            if not parts:
                continue
            ids.add(parts[0])
    return ids

def main():
    parser = argparse.ArgumentParser(description='Verify that output CSV user IDs match input TXT user IDs (original IDs).')
    parser.add_argument('--txt', required=True, help='Path to input TXT file used for evaluation')
    parser.add_argument('--csv', required=True, help='Path to output CSV results file')
    parser.add_argument('--meta', required=True, help='Path to metadata.json for original user mapping')
    args = parser.parse_args()

    if not os.path.isfile(args.txt):
        raise FileNotFoundError(args.txt)
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(args.csv)
    if not os.path.isfile(args.meta):
        raise FileNotFoundError(args.meta)

    metadata = load_metadata(args.meta)
    users_original = metadata.get('users', [])
    if not users_original:
        raise ValueError('metadata.json missing users list')

    txt_ids = users_from_txt(args.txt, users_original)
    csv_ids = users_from_csv(args.csv)

    missing_in_csv = txt_ids - csv_ids
    extra_in_csv = csv_ids - txt_ids

    print(f'TXT user count: {len(txt_ids)}')
    print(f'CSV user count: {len(csv_ids)}')
    if missing_in_csv:
        print(f'IDs missing in CSV ({len(missing_in_csv)}): sample -> {list(missing_in_csv)[:10]}')
    if extra_in_csv:
        print(f'Extra IDs in CSV ({len(extra_in_csv)}): sample -> {list(extra_in_csv)[:10]}')

    if not missing_in_csv and not extra_in_csv:
        print('SUCCESS: CSV IDs match TXT IDs.')
        exit(0)
    else:
        print('FAIL: Mismatch between CSV and TXT user ID sets.')
        exit(1)

if __name__ == '__main__':
    main()