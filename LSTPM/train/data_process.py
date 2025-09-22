import time
import os
import json
import argparse
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt
from collections import Counter


def entropy_spatial(sessions):
    locations = {}
    days = sorted(sessions.keys())
    for d in days:
        session = sessions[d]
        for s in session:
            if s[0] not in locations:
                locations[s[0]] = 1
            else:
                locations[s[0]] += 1
    frequency = np.array([locations[loc] for loc in locations])
    frequency = frequency / np.sum(frequency)
    entropy = - np.sum(frequency * np.log(frequency))
    return entropy


class DataFoursquare(object):
    def __init__(self, trace_min=0, global_visit=0, hour_gap=200, min_gap=0, session_min=1, session_max=10000,
                 sessions_min=1, train_split=0.8, embedding_len=50, data_path=None, save_path=None, metadata_json=None, secondary=False):
        tmp_path = "data/"
        # Input/output paths
        self.TWITTER_PATH = data_path or (tmp_path + 'foursquare/tweet_clean_new.txt')
        self.VENUES_PATH = tmp_path + 'foursquare/venues_all.txt'
        self.SAVE_PATH = tmp_path
        self.save_name = 'foursquare_cut_one_day'
        # If an explicit output path is passed, use that; else default to legacy path pattern
        self.output_path = save_path or (self.SAVE_PATH + self.save_name + '.pk')
        self.metadata_json = metadata_json
        self.secondary = secondary

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.words_original = []
        self.words_lens = []
        self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}
        self.vid_list_lookup = {}
        self.vid_lookup = {}
        self.pid_loc_lat = {}
        self.data_neural = {}

        # Preload unified mapping from metadata if provided
        if self.metadata_json:
            self.load_metadata()

    def load_metadata(self):
        if not self.metadata_json or not os.path.exists(self.metadata_json):
            raise FileNotFoundError(f"metadata.json not found: {self.metadata_json}")
        with open(self.metadata_json, 'r') as f:
            meta = json.load(f)
        pid_mapping = meta.get('pid_mapping', {})  # pid -> [lat, lon]
        if not pid_mapping:
            raise ValueError("metadata.json missing pid_mapping")
        # Build pid -> [lon, lat]
        self.pid_loc_lat = {}
        # Preserve insertion order of metadata for stable indices
        for pid, lat_lon in pid_mapping.items():
            lat, lon = lat_lon
            self.pid_loc_lat[pid] = [float(lon), float(lat)]
            if pid not in self.vid_list:
                vid = len(self.vid_list)
                self.vid_list[pid] = [vid, 0]
                self.vid_list_lookup[vid] = pid
                self.vid_lookup[vid] = [float(lon), float(lat)]




    # ############# 1. read trajectory data from twitters
    def load_trajectory_from_tweets(self):
        # Expected format from convert.py: tid\x01lat\x01lon\x01timestamp\x01pid
        with open(self.TWITTER_PATH, encoding='UTF-8') as fid:
            for i, line in enumerate(fid):
                parts = line.strip('\r\n').split('\x01')
                if len(parts) >= 5:
                    uid, lat, lon, tim, pid = parts[:5]
                else:
                    # Fallback for legacy format: uid, lon, lat, tim, pid
                    uid, lon, lat, tim, pid = parts
                if uid not in self.data:
                    self.data[uid] = [[pid, tim]]
                else:
                    self.data[uid].append([pid, tim])
                if pid not in self.venues:
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1

    # ########### 3.0 basically filter users based on visit length and other statistics
    def filter_users_by_length(self):
        uid_3 = [x for x in self.data if len(self.data[x]) > self.trace_len_min]
        xixi = [(x, len(self.data[x])) for x in uid_3]
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)
        pid_3 = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)
        pid_3 = dict(pid_pic3)

        session_len_list = []
        for u in pick3:
            uid = u[0]
            info = self.data[uid]
            xixi = Counter([x[0] for x in info])
            topk = Counter([x[0] for x in info]).most_common()
            topk1 = [x[0] for x in topk if x[1] > 1]
            sessions = {}
            for i, record in enumerate(info):
                poi, tmd = record
                try:
                    current_date = tmd.split(' ')[0]
                    tid = int(time.mktime(time.strptime(tmd, "%Y-%m-%d %H:%M:%S")))
                except Exception as e:
                    print('error:{}'.format(e))
                    continue
                sid = len(sessions)
                if poi not in pid_3 and poi not in topk1:
                    # if poi not in topk1:
                    continue
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    # if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max:
                    if last_date != current_date:
                        sessions[sid] = [record]
                    if (tid - last_tid) / 60 > self.min_gap:
                        sessions[sid - 1].append(record)
                    else:
                        pass
                last_tid = tid
                last_date = current_date
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            if len(sessions_filter) >= self.sessions_count_min:
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions}

        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count'] >= self.sessions_count_min]

    def filter_users_by_length_source(self):
        uid_3 = [x for x in self.data if len(self.data[x]) > self.trace_len_min]
        xixi = [(x, len(self.data[x])) for x in uid_3]
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)
        pid_3 = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)
        pid_3 = dict(pid_pic3)

        session_len_list = []
        for u in pick3:
            uid = u[0]
            info = self.data[uid]
            xixi = Counter([x[0] for x in info])
            topk = Counter([x[0] for x in info]).most_common()
            topk1 = [x[0] for x in topk if x[1] > 1]
            sessions = {}
            for i, record in enumerate(info):
                poi, tmd = record
                try:
                    tid = int(time.mktime(time.strptime(tmd, "%Y-%m-%d %H:%M:%S")))
                except Exception as e:
                    print('error:{}'.format(e))
                    continue
                sid = len(sessions)
                if poi not in pid_3 and poi not in topk1:
                    # if poi not in topk1:
                    continue
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max:
                        sessions[sid] = [record]
                    elif (tid - last_tid) / 60 > self.min_gap:
                        sessions[sid - 1].append(record)
                    else:
                        pass
                last_tid = tid
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            if len(sessions_filter) >= self.sessions_count_min:
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions}

        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count'] >= self.sessions_count_min]

    # ########### 4. build dictionary for users and location
    def build_users_locations_dict(self):
        for u in self.user_filter3:
            sessions = self.data_filter[u]['sessions']
            if u not in self.uid_list:
                self.uid_list[u] = [len(self.uid_list), len(sessions)]
            for sid in sessions:
                poi = [p[0] for p in sessions[sid]]
                for p in poi:
                    if p not in self.vid_list:
                        # When using metadata, keep vocabulary fixed to the unified mapping.
                        if not self.metadata_json:
                            self.vid_list_lookup[len(self.vid_list)] = p
                            self.vid_list[p] = [len(self.vid_list), 1]
                        else:
                            # unseen pid in metadata: skip (treated as 'unk' downstream)
                            continue
                    else:
                        self.vid_list[p][1] += 1

    # support for radius of gyration
    def load_venues(self):
        # If metadata provided, coordinates have already been loaded.
        if self.pid_loc_lat:
            return
        with open(self.TWITTER_PATH, 'r', encoding='UTF-8') as fid:
            for line in fid:
                parts = line.strip('\r\n').split('\x01')
                if len(parts) >= 5:
                    uid, lat, lon, tim, pid = parts[:5]
                else:
                    uid, lon, lat, tim, pid = parts
                try:
                    self.pid_loc_lat[pid] = [float(lon), float(lat)]
                except Exception:
                    continue

    def venues_lookup(self):
        for vid in self.vid_list_lookup:
            pid = self.vid_list_lookup[vid]
            lon_lat = self.pid_loc_lat[pid]
            self.vid_lookup[vid] = lon_lat

    # ########## 5.0 prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        timeStamp = (time.mktime(tm))
        # tid = tm.tm_hour
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return [timeStamp, tid]

    def prepare_neural_data(self):
        for u in self.uid_list:
            sessions = self.data_filter[u]['sessions']
            sessions_tran = {}
            sessions_id = []
            for sid in sessions:
                sessions_tran[sid] = [[self.vid_list[p[0]][0], p[1]] for p in
                                      sessions[sid]]
                sessions_id.append(sid)
            split_id = int(np.floor(self.train_split * len(sessions_id)))
            train_id = sessions_id[:split_id]
            test_id = sessions_id[split_id:]
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id])
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id])
            train_loc = {}
            for i in train_id:
                for sess in sessions_tran[i]:
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            # calculate entropy
            entropy = entropy_spatial(sessions)

            # calculate location ratio
            train_location = []
            for i in train_id:
                train_location.extend([s[0] for s in sessions[i]])
            train_location_set = set(train_location)
            test_location = []
            for i in test_id:
                test_location.extend([s[0] for s in sessions[i]])
            test_location_set = set(test_location)
            whole_location = train_location_set | test_location_set
            if len(whole_location) == 0:
                location_ratio = 0.0
            else:
                test_unique = whole_location - train_location_set
                location_ratio = len(test_unique) / len(whole_location)

            # calculate radius of gyration
            lon_lat = []
            for pid in train_location:
                try:
                    lon_lat.append(self.pid_loc_lat[pid])
                except Exception:
                    continue
            if len(lon_lat) == 0:
                rg = 0.0
            else:
                lon_lat = np.array(lon_lat, dtype=np.float32).reshape(-1, 2)
                center = lon_lat.mean(axis=0)
                diffs = lon_lat - center
                rg = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))

            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id,
                                                     'pred_len': pred_len, 'valid_len': valid_len,
                                                     'train_loc': train_loc, 'explore': location_ratio,
                                                     'entropy': entropy, 'rg': rg}

    # ############# 6. save variables
    def get_parameters(self):
        parameters = {}
        parameters['TWITTER_PATH'] = self.TWITTER_PATH
        parameters['SAVE_PATH'] = self.SAVE_PATH

        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['train_split'] = self.train_split

        return parameters

    def save_variables(self):
        foursquare_dataset = {'data_neural': self.data_neural, 'vid_list': self.vid_list, 'uid_list': self.uid_list,
                              'parameters': self.get_parameters(), 'data_filter': self.data_filter,
                              'vid_lookup': self.vid_lookup}
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        pickle.dump(foursquare_dataset, open(self.output_path, 'wb'))
        print('saved to ' + self.output_path)


def geodistance(lng1, lat1, lng2, lat2):
    print('calculating geod')
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance/1000, 3)
    return distance


def build_unified_distance_from_metadata(metadata_path, out_distance_path):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    pid_mapping = meta.get('pid_mapping', {})  # pid -> [lat, lon]
    if not pid_mapping:
        raise ValueError("metadata.json missing pid_mapping; cannot build distance matrix")
    pids = list(pid_mapping.keys())
    n = len(pids)
    print(f"Building unified distance matrix for {n} POIs from metadata...")
    # Prepare coordinate arrays (lon, lat) in radians
    lons = np.array([pid_mapping[p][1] for p in pids], dtype=np.float64)
    lats = np.array([pid_mapping[p][0] for p in pids], dtype=np.float64)
    lon_r = np.radians(lons)[:, None]
    lat_r = np.radians(lats)[:, None]
    # Broadcast to NxN
    dlon = lon_r - lon_r.T
    dlat = lat_r - lat_r.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * 6371.0  # kilometers
    dist = dist.astype(np.float32)
    # Apply minimum distance of 1 km for non-zero entries
    dist[dist < 1.0] = 1.0
    dist[np.diag_indices_from(dist)] = 0.0
    # Allocate with padding for 'unk' at index 0
    mat = np.zeros((n + 1, n + 1), dtype=np.float32)
    mat[1:, 1:] = dist
    os.makedirs(os.path.dirname(out_distance_path), exist_ok=True)
    with open(out_distance_path, 'wb') as fh:
        pickle.dump(mat, fh)
    print(f"Saved unified distance matrix to {out_distance_path} (size {(n+1)}x{(n+1)})")


def process_directory(in_dir, out_dir, train_filename, train_split_ratio=0.8, metadata_json_path=None, distance_out_path=None):
    print(f"Processing all .txt files from directory: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    # Allow caller to specify where metadata.json lives (e.g., run top dir)
    metadata_path = metadata_json_path if metadata_json_path else os.path.join(in_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Required metadata.json not found at {metadata_path}")
    # Build unified distance once; default next to out_dir unless overridden
    distance_out = distance_out_path if distance_out_path else os.path.join(out_dir, 'distance.pkl')
    try:
        build_unified_distance_from_metadata(metadata_path, distance_out)
    except Exception as e:
        print(f"Warning: distance matrix not created: {e}")

    txt_files = [f for f in os.listdir(in_dir) if f.endswith('.txt')]
    if not txt_files:
        print("No .txt files found in the input directory.")
        return
    if train_filename not in txt_files:
        raise FileNotFoundError(f"Train file '{train_filename}' not found in {in_dir}")

    for txt in txt_files:
        in_path = os.path.join(in_dir, txt)
        out_path = os.path.join(out_dir, os.path.splitext(txt)[0] + '.pk')
        current_split = train_split_ratio if txt == train_filename else 1.0
        is_secondary = txt != train_filename
        print(f"\n--- Processing {txt} (train_split={current_split}) -> {out_path}")
        dg = DataFoursquare(
            data_path=in_path,
            save_path=out_path,
            metadata_json=metadata_path,
            secondary=is_secondary,
            trace_min=0, global_visit=0, hour_gap=200, min_gap=0,
            session_max=10000, session_min=1, sessions_min=1,
            train_split=current_split
        )
        dg.load_trajectory_from_tweets()
        dg.filter_users_by_length()
        dg.build_users_locations_dict()
        dg.load_venues()
        dg.venues_lookup()
        dg.prepare_neural_data()
        dg.save_variables()


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess trajectory data for LSTPM.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Single file mode
    pf = subparsers.add_parser('file', help='Process a single DeepMove-formatted TXT file.')
    pf.add_argument('--data_path', required=True, help='Path to input TXT file (tid\u0001lat\u0001lon\u0001timestamp\u0001pid)')
    pf.add_argument('--save_path', required=True, help='Path to output .pk file')
    pf.add_argument('--metadata_json', required=True, help='Path to unified metadata.json (from convert.py)')
    pf.add_argument('--train_split', type=float, default=0.8, help='Train/test ratio (ignored if secondary)')
    pf.add_argument('--secondary', action='store_true', help='Mark this file as secondary (pure test)')

    # Directory mode
    pdp = subparsers.add_parser('dir', help='Process all TXT files in a directory with shared metadata.')
    pdp.add_argument('--in_dir', required=True, help='Directory with TXT files and metadata.json')
    pdp.add_argument('--out_dir', required=True, help='Output directory for .pk files')
    pdp.add_argument('--training_set_name', required=True, help='Filename of the training TXT within in_dir')
    pdp.add_argument('--train_split', type=float, default=0.8, help='Train/test ratio for the training file')
    pdp.add_argument('--metadata_json', type=str, default=None, help='Explicit path to metadata.json (e.g., run top)')
    pdp.add_argument('--distance_out', type=str, default=None, help='Explicit path for distance.pkl (e.g., run_top/distance.pkl)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.command == 'file':
        dg = DataFoursquare(
            data_path=args.data_path,
            save_path=args.save_path,
            metadata_json=args.metadata_json,
            secondary=args.secondary,
            train_split=(0.0 if args.secondary else args.train_split),
            trace_min=0, global_visit=0, hour_gap=20000, min_gap=0,
            session_max=10000, session_min=1, sessions_min=1,
        )
        dg.load_trajectory_from_tweets()
        dg.filter_users_by_length()
        dg.build_users_locations_dict()
        dg.load_venues()
        dg.venues_lookup()
        dg.prepare_neural_data()
        dg.save_variables()
        # Also build a distance matrix next to the output pk (using metadata)
        try:
            build_unified_distance_from_metadata(args.metadata_json, os.path.join(os.path.dirname(args.save_path), 'distance.pkl'))
        except Exception as e:
            print(f"Warning: distance matrix not created: {e}")
    elif args.command == 'dir':
        process_directory(
            args.in_dir,
            args.out_dir,
            args.training_set_name,
            args.train_split,
            metadata_json_path=args.metadata_json,
            distance_out_path=args.distance_out,
        )
