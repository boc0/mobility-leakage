import argparse
import os
import shutil
from convert import convert_directory
from data_process import process_directory as process_txt_dir


def preprocess(in_dir: str, out_dir: str, training_set_name: str):
    """
    Orchestrates preprocessing for LSTPM:
    1) Convert all CSVs in in_dir to DeepMove TXT format under out_dir/txts,
       and write unified metadata.json into out_dir (run top directory).
    2) Run data_process on the TXT directory to produce PK files under out_dir/preprocessed,
       using the metadata.json from out_dir. Also ensure distance.pkl is saved at out_dir/distance.pkl.
    """
    # Ensure top-level run directory exists
    os.makedirs(out_dir, exist_ok=True)

    # 1) Convert CSVs -> TXT; convert_directory writes metadata.json into the out_dir argument
    txts_dir = os.path.join(out_dir, 'txts')
    os.makedirs(txts_dir, exist_ok=True)
    convert_directory(in_dir, out_dir=txts_dir)

    # Move metadata.json from txts_dir to run top (out_dir) if present
    src_meta = os.path.join(txts_dir, 'metadata.json')
    dst_meta = os.path.join(out_dir, 'metadata.json')
    if os.path.exists(src_meta):
        shutil.move(src_meta, dst_meta)
    else:
        # If convert_directory already placed it at run top, fine; otherwise error
        if not os.path.exists(dst_meta):
            raise FileNotFoundError('metadata.json not found after convert step')

    # 2) TXT -> PK using data_process with explicit metadata path
    pre_dir = os.path.join(out_dir, 'preprocessed')
    os.makedirs(pre_dir, exist_ok=True)

    # Normalize training_set_name to .txt and resolve robustly if not exact
    base, ext = os.path.splitext(training_set_name)
    train_txt = base + '.txt' if ext == '' else base + ext
    # If exact name not present, try to find a unique partial match
    available_txts = [f for f in os.listdir(txts_dir) if f.endswith('.txt')]
    if train_txt not in available_txts:
        # partial match by substring on base name
        matches = [f for f in available_txts if base in os.path.splitext(f)[0]]
        if len(matches) == 1:
            train_txt = matches[0]
        else:
            raise FileNotFoundError(
                f"Training TXT '{train_txt}' not found in {txts_dir}. Available: {available_txts}"
            )

    # Build distance at run top
    distance_out = os.path.join(out_dir, 'distance.pkl')

    process_txt_dir(
        in_dir=txts_dir,
        out_dir=pre_dir,
        train_filename=train_txt,
        train_split_ratio=0.8,
        metadata_json_path=dst_meta,
        distance_out_path=distance_out,
    )


def main():
    parser = argparse.ArgumentParser(description='LSTPM preprocess: CSV dir -> TXT (+metadata at run top) -> PKs (+distance at run top)')
    parser.add_argument('--in_dir', required=True, help='Input directory with CSV files')
    parser.add_argument('--training_set_name', required=True, help='Base name (without extension) of the training CSV/TXT')
    parser.add_argument('--out_dir', required=True, help='Run top directory (e.g., run0)')
    args = parser.parse_args()

    preprocess(args.in_dir, args.out_dir, args.training_set_name)


if __name__ == '__main__':
    main()
