import argparse
import os
import shutil
from convert import convert_directory
from sparse_traces import process_directory as process_txt_dir


def preprocess(in_dir: str, out_dir: str, training_set_name: str):
	"""
	Orchestrates preprocessing:
	1) Convert all CSVs in in_dir to DeepMove TXT format under out_dir/txts,
	   and write unified metadata.json into out_dir (run top directory).
	2) Run sparse_traces on the TXT directory to produce PK files under out_dir/preprocessed,
	   using the metadata.json from out_dir.
	"""
	# Ensure top-level run directory exists
	os.makedirs(out_dir, exist_ok=True)

	# 1) Convert CSVs -> TXT with unified metadata.json in out_dir
	txts_dir = os.path.join(out_dir, 'txts')
	os.makedirs(txts_dir, exist_ok=True)
	# convert_directory writes metadata.json into out_dir param; we want it at txts_dir then move to out_dir
	convert_directory(in_dir, out_dir=txts_dir)

	# Move metadata.json from txts_dir to out_dir if present
	src_meta = os.path.join(txts_dir, 'metadata.json')
	dst_meta = os.path.join(out_dir, 'metadata.json')
	if os.path.exists(src_meta):
		shutil.move(src_meta, dst_meta)
	else:
		# If convert_directory already placed it at out_dir, keep going
		if not os.path.exists(dst_meta):
			raise FileNotFoundError('metadata.json not found after convert step')

	# 2) TXT -> PK using sparse_traces with explicit metadata path
	pre_dir = os.path.join(out_dir, 'preprocessed')
	os.makedirs(pre_dir, exist_ok=True)
	if len(os.path.splitext(training_set_name)) > 1:
		extension = os.path.splitext(training_set_name)[1]
		train_txt = f"{os.path.splitext(training_set_name)[0]}.txt"
	else:
		train_txt = f"{training_set_name}.txt"
	

	process_txt_dir(
		in_dir=txts_dir,
		out_dir=pre_dir,
		train_filename=train_txt,
		train_split_ratio=0.8,
		metadata_json_path=dst_meta,
	)


def main():
	parser = argparse.ArgumentParser(description='End-to-end preprocess: CSV dir -> TXT (+metadata) -> PK')
	parser.add_argument('--in_dir', required=True, help='Input directory with CSV files')
	parser.add_argument('--training_set_name', required=True, help='Base name (without extension) of the training CSV/TXT')
	parser.add_argument('--out_dir', required=True, help='Run top directory (e.g., run0)')
	args = parser.parse_args()

	preprocess(args.in_dir, args.out_dir, args.training_set_name)


if __name__ == '__main__':
	main()