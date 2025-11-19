import argparse
import csv
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess
import sys
from collections import defaultdict
from scipy.sparse import coo_matrix

"""Graph-Flashback preprocessing

Converts a directory of DeepMove-style CSV trajectory files into the
Graph-Flashback raw TXT format (tab-separated, no header):

user_id<TAB>timestamp<TAB>lat<TAB>lon<TAB>poi_id

Steps:
1. Scan all input CSVs (expect header with at least: tid,timestamp,lat,lon).
2. Derive user identifiers from the `tid` column (configurable).
3. Enumerate global unique (lat, lon) pairs -> assign sequential poi_ids.
4. Write one output TXT per input file preserving original row order.
5. Emit metadata.json with pid_mapping and users list (similar to DeepMove expectations).

Notes:
- Assumes consistent 30-minute sampling but does not enforce.
- Multiple trajectories (different tid values) mapping to same user are allowed; if
  `--user_mode tid_prefix` is used, the user id is the substring before the first '_'.
- If coordinates repeat they map to the same poi_id globally across all files.
- Output files keep original file base name but with .txt extension.
"""

def normalize_timestamp(ts: str) -> str:
	"""Normalize various timestamp strings to ISO8601 'YYYY-MM-DDTHH:MM:SSZ'.
	Accepts common forms like 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DDTHH:MM:SS',
	and variants with '/' separators or trailing 'Z'. Falls back to a
	simple replace if parsing fails.
	"""
	if not ts:
		return ts
	s = ts.strip()
	# Already in desired format
	if 'T' in s and s.endswith('Z'):
		return s
	# Remove trailing Z for parsing if present
	s_noz = s[:-1] if s.endswith('Z') else s
	fmts = [
		'%Y-%m-%d %H:%M:%S',
		'%Y-%m-%dT%H:%M:%S',
		'%Y/%m/%d %H:%M:%S',
		'%Y/%m/%dT%H:%M:%S',
	]
	for fmt in fmts:
		try:
			dt = datetime.strptime(s_noz, fmt)
			return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
		except ValueError:
			pass
	# Last resort: simple transformation
	s_basic = s_noz.replace(' ', 'T')
	if len(s_basic) >= 19:  # naive guard for 'YYYY-MM-DDTHH:MM:SS'
		return s_basic[:19] + 'Z'
	return s

def gather_rows(path: str) -> List[Dict[str, str]]:
	"""Read a CSV file returning list of row dicts.
	Expected columns: tid,timestamp,lat,lon (case-insensitive variants accepted).
	Extra columns are ignored.
	"""
	rows: List[Dict[str, str]] = []
	with open(path, 'r', newline='') as f:
		sample = f.read(1024)
		f.seek(0)
		# Detect delimiter (comma or tab); default comma
		delimiter = '\t' if ('\t' in sample and ',' not in sample.split('\n')[0]) else ','
		reader = csv.DictReader(f, delimiter=delimiter)
		required = {'tid', 'timestamp', 'lat', 'lon'}
		header_lower = {h.lower(): h for h in reader.fieldnames or []}
		if not required.issubset(set(header_lower.keys())):
			raise ValueError(f"File {path} missing required columns {required}. Found: {reader.fieldnames}")
		for r in reader:
			# Normalize keys to lower
			row_norm = {k.lower(): v for k, v in r.items()}
			rows.append(row_norm)
	return rows

def derive_user_id(tid: str, mode: str) -> str:
	"""Derive user identifier from trajectory id according to mode.
	modes:
	  tid_full  -> use the entire tid string
	  tid_prefix -> substring before first '_' (fallback to full if '_' absent)
	"""
	tid = tid.strip()
	if mode == 'tid_prefix':
		return tid.split('_')[0]
	return tid

def preprocess_directory(in_dir: str, out_dir: str, user_mode: str = 'tid_prefix') -> None:
	if not os.path.isdir(in_dir):
		raise FileNotFoundError(f"Input directory not found: {in_dir}")
	os.makedirs(out_dir, exist_ok=True)

	input_files = [f for f in os.listdir(in_dir) if f.lower().endswith('.csv')]
	if not input_files:
		raise FileNotFoundError(f"No CSV files found in {in_dir}")

	# First pass: gather all rows and collect unique users & poi coordinates
	all_rows_per_file: Dict[str, List[Dict[str, str]]] = {}
	poi_map: Dict[Tuple[float, float], int] = {}
	users_set: set = set()

	for fname in sorted(input_files):
		full_path = os.path.join(in_dir, fname)
		rows = gather_rows(full_path)
		all_rows_per_file[fname] = rows
		for r in rows:
			try:
				lat = float(r['lat'])
				lon = float(r['lon'])
			except ValueError:
				# Skip malformed coordinate row
				continue
			coord = (round(lat, 7), round(lon, 7))  # normalize precision
			if coord not in poi_map:
				poi_map[coord] = len(poi_map)
			uid = derive_user_id(r['tid'], user_mode)
			users_set.add(uid)

	# Prepare metadata and mappings
	pid_mapping = {str(pid): [coord[0], coord[1]] for coord, pid in poi_map.items()}
	users_list = sorted(users_set)
	user_to_id: Dict[str, int] = {u: i for i, u in enumerate(users_list)}
	metadata = {
		'pid_mapping': pid_mapping,
		'users': users_list,
		'num_users': len(users_list),
		'num_pois': len(pid_mapping),
		'user_mode': user_mode,
		'generated_at': datetime.utcnow().isoformat() + 'Z'
	}
	meta_path = os.path.join(out_dir, 'metadata.json')
	with open(meta_path, 'w') as fh:
		json.dump(metadata, fh, indent=2)

	# Second pass: write converted files and build union file
	union_path = os.path.join(out_dir, 'union.txt')
	union_fh = open(union_path, 'w')
	for fname, rows in all_rows_per_file.items():
		out_name = os.path.splitext(fname)[0] + '.txt'
		out_path = os.path.join(out_dir, out_name)
		with open(out_path, 'w') as out_f:
			for r in rows:
				try:
					lat = float(r['lat']); lon = float(r['lon'])
				except ValueError:
					continue
				coord = (round(lat, 7), round(lon, 7))
				poi_id = poi_map[coord]
				uid_str = derive_user_id(r['tid'], user_mode)
				uid = user_to_id.get(uid_str, None)
				if uid is None:
					# Should not happen, but guard anyway
					continue
				timestamp = normalize_timestamp(r['timestamp'])
				# Write tab-separated line
				line = f"{uid}\t{timestamp}\t{lat}\t{lon}\t{poi_id}\n"
				out_f.write(line)
				union_fh.write(line)
	union_fh.close()

	print(f"Preprocessing complete. Wrote {len(all_rows_per_file)} files and metadata.json with {len(poi_map)} POIs.")

	# Run KGE triplet generation and refine on the union file
	try:
		from KGE.constant import DATA_NAME, SCHEME
	except Exception:
		DATA_NAME, SCHEME = 'training_set', 2

	# Ensure an empty friendship file exists under Graph-Flashback/data
	repo_root = os.path.dirname(__file__)
	data_dir = os.path.join(repo_root, 'data')
	os.makedirs(data_dir, exist_ok=True)
	friend_path = os.path.join(data_dir, 'friendship.txt')
	if not os.path.exists(friend_path):
		with open(friend_path, 'w') as f:
			pass

	kge_dir = os.path.join(repo_root, 'KGE')
	# Use absolute paths for external module calls
	union_abs = os.path.abspath(union_path)
	# Pass only the filename for friendship, Setting will prefix ./data/
	friend_name = os.path.basename(friend_path)
	# Call generate_triplet
	cmd_gen = [sys.executable, '-m', 'KGE.generate_triplet', '--dataset', union_abs, '--friendship', friend_name]
	print('Running:', ' '.join(cmd_gen))
	res = subprocess.run(cmd_gen, cwd=repo_root)
	if res.returncode != 0:
		print('[Warn] KGE.generate_triplet failed. Please check logs.')
		return
	# Call refine
	cmd_ref = [sys.executable, '-m', 'KGE.refine']
	print('Running:', ' '.join(cmd_ref))
	res2 = subprocess.run(cmd_ref, cwd=repo_root)
	if res2.returncode != 0:
		print('[Warn] KGE.refine failed. Please check logs.')
		return

	# Overwrite final_* with new_final_* for convenience
	dataset_dir = os.path.join(kge_dir, 'dataset', DATA_NAME, f'{DATA_NAME}_scheme{SCHEME}')
	new_train = os.path.join(dataset_dir, 'new_final_train_triplets.txt')
	new_test = os.path.join(dataset_dir, 'new_final_test_triplets.txt')
	final_train = os.path.join(dataset_dir, 'final_train_triplets.txt')
	final_test = os.path.join(dataset_dir, 'final_test_triplets.txt')
	try:
		if os.path.exists(new_train):
			with open(new_train, 'r') as src, open(final_train, 'w') as dst:
				dst.write(src.read())
		if os.path.exists(new_test):
			with open(new_test, 'r') as src, open(final_test, 'w') as dst:
				dst.write(src.read())
		print('Triplet refine completed and final_* files updated.')
	except Exception as e:
		print(f'[Warn] Unable to update final triplet files: {e}')

	# Copy key triplet artifacts into output directory for self-contained training
	for fname in ['final_train_triplets.txt', 'final_test_triplets.txt', 'entity2id.txt', 'relation2id.txt']:
		src = os.path.join(dataset_dir, fname)
		if os.path.exists(src):
			dst = os.path.join(out_dir, fname)
			try:
				with open(src, 'r') as fsrc, open(dst, 'w') as fdst:
					fdst.write(fsrc.read())
				print(f'Copied {fname} to output directory.')
			except Exception as e:
				print(f'[Warn] Failed to copy {fname}: {e}')

	# Build simple transition (POI->POI) and interaction (user->POI) graphs from union data
	print('Building simple graphs from union.txt ...')
	# Read union for sequences per user ordered by timestamp
	user_checkins: Dict[int, List[Tuple[datetime, int]]] = defaultdict(list)
	with open(union_path, 'r') as uf:
		for line in uf:
			parts = line.strip().split('\t')
			if len(parts) != 5:
				continue
			uid, ts, lat, lon, pid = parts
			try:
				dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')
				user_checkins[int(uid)].append((dt, int(pid)))
			except Exception:
				continue
	# Sort each user's list chronologically
	for u in user_checkins:
		user_checkins[u].sort(key=lambda x: x[0])

	loc_count = len(poi_map)
	user_count = len(user_to_id)
	# Transition counts
	trans_counts = defaultdict(int)
	for u, seq in user_checkins.items():
		for i in range(1, len(seq)):
			prev_loc = seq[i-1][1]
			cur_loc = seq[i][1]
			trans_counts[(prev_loc, cur_loc)] += 1

	if trans_counts:
		rows = []
		cols = []
		data = []
		for (src_loc, dst_loc), cnt in trans_counts.items():
			rows.append(src_loc)
			cols.append(dst_loc)
			data.append(float(cnt))
		transition_mx = coo_matrix((data, (rows, cols)), shape=(loc_count, loc_count))
	else:
		transition_mx = coo_matrix((loc_count, loc_count))

	# Interaction counts (user -> poi)
	inter_rows = []
	inter_cols = []
	inter_data = []
	visit_counts = defaultdict(int)
	for u, seq in user_checkins.items():
		for _, loc in seq:
			visit_counts[(u, loc)] += 1
	for (u, loc), cnt in visit_counts.items():
		inter_rows.append(u)
		inter_cols.append(loc)
		inter_data.append(float(cnt))
	interact_mx = coo_matrix((inter_data, (inter_rows, inter_cols)), shape=(user_count, loc_count))

	# Persist graphs
	trans_loc_file = os.path.join(out_dir, 'trans_loc.pkl')
	trans_interact_file = os.path.join(out_dir, 'trans_interact.pkl')
	with open(trans_loc_file, 'wb') as f:
		pickle.dump(transition_mx, f)
	with open(trans_interact_file, 'wb') as f:
		pickle.dump(interact_mx, f)
	print(f'Wrote graphs: {trans_loc_file}, {trans_interact_file}')

	# Augment metadata with graph file paths
	try:
		with open(meta_path, 'r') as mfh:
			md = json.load(mfh)
		md['trans_loc_file'] = trans_loc_file
		md['trans_interact_file'] = trans_interact_file
		with open(meta_path, 'w') as mfh:
			json.dump(md, mfh, indent=2)
	except Exception as e:
		print(f'[Warn] Could not augment metadata with graph paths: {e}')

	# Build per-file graphs with union shapes for targeted training/eval
	try:
		for fname in sorted(all_rows_per_file.keys()):
			out_name = os.path.splitext(fname)[0] + '.txt'
			per_file_path = os.path.join(out_dir, out_name)
			# collect per-file sequences
			pf_user_checkins: Dict[int, List[int]] = defaultdict(list)
			with open(per_file_path, 'r') as pf:
				for line in pf:
					parts = line.strip().split('\t')
					if len(parts) != 5:
						continue
					uid, ts, lat, lon, pid = parts
					try:
						pf_user_checkins[int(uid)].append(int(pid))
					except Exception:
						continue
			# Transition counts per-file
			rows = []
			cols = []
			data = []
			for u, seq in pf_user_checkins.items():
				for i in range(1, len(seq)):
					rows.append(seq[i-1])
					cols.append(seq[i])
					data.append(1.0)
			pf_transition = coo_matrix((data, (rows, cols)), shape=(loc_count, loc_count))

			# Interaction counts per-file
			rows = []
			cols = []
			data = []
			for u, seq in pf_user_checkins.items():
				count_by_loc = defaultdict(int)
				for loc in seq:
					count_by_loc[loc] += 1
				for loc, cnt in count_by_loc.items():
					rows.append(int(u))
					cols.append(loc)
					data.append(float(cnt))
			pf_interact = coo_matrix((data, (rows, cols)), shape=(user_count, loc_count))

			# Persist per-file graphs
			pf_base = os.path.splitext(out_name)[0]
			pf_trans_loc = os.path.join(out_dir, f'trans_loc_{pf_base}.pkl')
			pf_trans_interact = os.path.join(out_dir, f'trans_interact_{pf_base}.pkl')
			with open(pf_trans_loc, 'wb') as f:
				pickle.dump(pf_transition, f)
			with open(pf_trans_interact, 'wb') as f:
				pickle.dump(pf_interact, f)
			print(f'Per-file graphs written for {pf_base}')
	except Exception as e:
		print(f'[Warn] Failed to build per-file graphs: {e}')
	# delete union file
	# if os.path.exists(union_path):
	# 	os.remove(union_path)

def cli():
	parser = argparse.ArgumentParser(description='Graph-Flashback preprocess: CSV -> raw TXT + metadata.json')
	parser.add_argument('--in_dir', required=True, help='Input directory containing CSV files')
	parser.add_argument('--out_dir', required=True, help='Output directory for TXT files and metadata.json')
	parser.add_argument('--user_mode', choices=['tid_full', 'tid_prefix'], default='tid_full', help='How to derive user id from tid column')
	args = parser.parse_args()
	preprocess_directory(args.in_dir, args.out_dir, user_mode=args.user_mode)

if __name__ == '__main__':
	cli()

