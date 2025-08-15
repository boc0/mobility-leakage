import time
import argparse
import numpy as np
import pickle as pickle
from collections import Counter
import json
import os


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
    def __init__(self, data_path, save_path='data/foursquare', trace_min=1, global_visit=1, hour_gap=72, min_gap=10, session_min=2, session_max=100,
                 sessions_min=1, train_split=0.8, embedding_len=50, secondary=False, metadata_json=None):
        self.secondary = secondary
        self.metadata_json = metadata_json
        tmp_path = "data/"
        # self.TWITTER_PATH = tmp_path + 'foursquare/tweets_clean.txt'
        self.TWITTER_PATH = data_path
        # self.VENUES_PATH = tmp_path + 'foursquare/venues_all.txt'
        self.SAVE_PATH = save_path

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

        if self.metadata_json:
            self.load_metadata()

    def load_metadata(self):
        """Load pid_mapping and users from metadata.json."""
        print(f"Loading metadata from {self.metadata_json}")
        with open(self.metadata_json, 'r') as f:
            meta = json.load(f)
        # pid_mapping is pid -> [lat, lon]
        # we need pid -> [vid, count] and pid -> [lon, lat]
        pid_mapping = meta.get('pid_mapping', {})
        self.pid_loc_lat = {pid: [lon, lat] for pid, (lat, lon) in pid_mapping.items()}
        # Pre-populate vid_list from the metadata
        for i, pid in enumerate(pid_mapping.keys()):
            # Assign a new vid, initialize count to 0. Count will be updated later.
            self.vid_list[pid] = [i + 1, 0] # +1 to reserve 0 for 'unk'
        print(f"Loaded {len(self.vid_list)} locations from metadata.")

    # ############# 1. read trajectory data from twitters
    def load_trajectory_from_tweets(self):
        with open(self.TWITTER_PATH) as fid:
            for i, line in enumerate(fid):
                try:
                    uid, _, _, tim, pid = line.strip('\r\n').split('')
                except:
                    print(f"Error parsing line {i}: {line.strip()}")
                    continue
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
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)
        pid_3 = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)
        pid_3 = dict(pid_pic3)

        session_len_list = []
        for u in pick3:
            uid = u[0]
            info = self.data[uid]
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
                        # This should not happen if metadata is complete
                        self.vid_list_lookup[len(self.vid_list)] = p
                        self.vid_list[p] = [len(self.vid_list), 1]
                    else:
                        self.vid_list[p][1] += 1

    # support for radius of gyration
    def load_venues(self):
        with open(self.TWITTER_PATH, 'r') as fid:
            for line in fid:
                uid, lon, lat, tim, pid = line.strip('\r\n').split('')
                self.pid_loc_lat[pid] = [float(lon), float(lat)]

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
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid

    def prepare_neural_data(self):
        # ---- SESSION-LEVEL train/test split (users are in both sets) ----
        for u in sorted(self.data_filter.keys()):
            sessions = self.data_filter[u]['sessions']
            sessions_tran = {}
            sessions_id = list(sessions.keys())
            for sid in sessions_id:
                sessions_tran[sid] = [[self.vid_list[p[0]][0], self.tid_list_48(p[1])]
                                       for p in sessions[sid]]
            # split user's sessions into train and test
            split_idx = int(np.floor(self.train_split * len(sessions_id)))
            train_id = sessions_id[:split_idx]
            test_id = sessions_id[split_idx:]
            # compute lengths
            pred_len = sum(len(sessions_tran[i]) - 1 for i in train_id)
            valid_len = sum(len(sessions_tran[i]) - 1 for i in test_id) if test_id else 0
            # collect train location frequencies
            train_loc = {}
            for i in train_id:
                for loc, _ in sessions_tran[i]:
                    train_loc[loc] = train_loc.get(loc, 0) + 1
            # entropy
            entropy = entropy_spatial(sessions)
            # location ratio
            train_locs = [p for i in train_id for p, _ in sessions[i]]
            test_locs = [p for i in test_id for p, _ in sessions[i]]
            whole = set(train_locs) | set(test_locs)
            test_unique = whole - set(train_locs)
            location_ratio = len(test_unique) / len(whole) if whole else 0.0
            # radius of gyration data (not used here)
            # store data
            self.data_neural[self.uid_list[u][0]] = {
                'sessions': sessions_tran,
                'train': train_id,
                'test': test_id,
                'pred_len': pred_len,
                'valid_len': valid_len,
                'train_loc': train_loc,
                'explore': location_ratio,
                'entropy': entropy
            }
        print('final users:{}'.format(len(self.data_neural)))

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
        # pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name + '.pk', 'wb'))
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH, 'wb'))


def process_directory(in_dir, out_dir, train_filename, train_split_ratio=0.8):
    """
    Processes all .txt files in a directory using a shared metadata.json,
    and outputs them as .pk files to the output directory.
    """
    print(f"Processing all .txt files from directory: {in_dir}")
    os.makedirs(out_dir, exist_ok=True)

    metadata_path = os.path.join(in_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Required metadata.json not found in input directory: {in_dir}")

    txt_files = [f for f in os.listdir(in_dir) if f.endswith('.txt')]
    if not txt_files:
        print("No .txt files found in the input directory.")
        return

    if train_filename not in txt_files:
        raise FileNotFoundError(f"Specified train file '{train_filename}' not found in input directory: {in_dir}")

    for txt_file in txt_files:
        in_path = os.path.join(in_dir, txt_file)
        out_filename = os.path.splitext(txt_file)[0] + '.pk'
        out_path = os.path.join(out_dir, out_filename)
        print(f"\n--- Processing {in_path} -> {out_path} ---")

        # Determine if this is a training or testing/secondary file
        is_secondary = 'secondary' in txt_file.lower() or 'test' in txt_file.lower()

        # Use the specified split ratio for the training file, 1.0 for all others.
        current_train_split = train_split_ratio if txt_file == train_filename else 1.0
        print(f"Using train_split: {current_train_split}")

        # Use default non-filtering parameters
        data_generator = DataFoursquare(
            data_path=in_path,
            save_path=out_path,
            metadata_json=metadata_path,
            secondary=is_secondary,
            trace_min=0, global_visit=0, hour_gap=200, min_gap=0,
            session_max=10000, session_min=1, sessions_min=1,
            train_split=current_train_split
        )

        data_generator.load_trajectory_from_tweets()
        data_generator.filter_users_by_length()
        data_generator.build_users_locations_dict()
        data_generator.venues_lookup()
        data_generator.prepare_neural_data()
        data_generator.save_variables()

    print("\nDirectory processing complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess trajectory data for DeepMove.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Single file command ---
    parser_file = subparsers.add_parser('file', help='Process a single trajectory file.')
    parser_file.add_argument('--data_path', type=str, required=True,
                        help="path to the input trajectory data")
    parser_file.add_argument('--save_path', type=str, required=True,
                        help="name of the saved dataset file")
    parser_file.add_argument('--metadata_json', type=str, default=None,
                        help="path to metadata json file to load existing pid mapping")
    parser_file.add_argument('--secondary', action='store_true',
                        help="set for the secondary dataset (what we will compute perplexity on)")
    # Filtering arguments
    parser_file.add_argument('--trace_min',   type=int, default=0, help="raw trace length filter threshold (0=keep all)")
    parser_file.add_argument('--global_visit',type=int, default=0, help="location global visit threshold (0=keep all)")
    parser_file.add_argument('--hour_gap',    type=int, default=200, help="max hours between points before new session")
    parser_file.add_argument('--min_gap',     type=int, default=0, help="min minutes between points to stay in session")
    parser_file.add_argument('--session_max', type=int, default=10000, help="max points per session")
    parser_file.add_argument('--session_min', type=int, default=1, help="min points per session")
    parser_file.add_argument('--sessions_min',type=int, default=1, help="min sessions per user")
    parser_file.add_argument('--train_split', type=float, default=0.8, help="train/test ratio for user splitting")

    # --- Directory command ---
    parser_dir = subparsers.add_parser('dir', help='Process all .txt files in a directory.')
    parser_dir.add_argument('--in_dir', type=str, required=True, help='Input directory path with .txt files and metadata.json')
    parser_dir.add_argument('--out_dir', type=str, default='preprocessed_pk', help='Output directory for .pk files')
    parser_dir.add_argument('--train', type=str, required=True, help='The name of the training file within the input directory.')
    parser_dir.add_argument('--train_split', type=float, default=0.8, help="train/test ratio for the training file")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'file':
        data_generator = DataFoursquare(data_path=args.data_path, save_path=args.save_path, trace_min=args.trace_min, global_visit=args.global_visit,
                                        hour_gap=args.hour_gap, min_gap=args.min_gap,
                                        session_min=args.session_min, session_max=args.session_max,
                                        sessions_min=args.sessions_min, train_split=args.train_split,
                                        secondary=args.secondary, metadata_json=args.metadata_json)
        parameters = data_generator.get_parameters()
        print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
        print('############START PROCESSING:')
        data_generator.load_trajectory_from_tweets()
        data_generator.filter_users_by_length()
        data_generator.build_users_locations_dict()
        if not args.metadata_json:
            data_generator.load_venues()
        data_generator.venues_lookup()
        data_generator.prepare_neural_data()
        data_generator.save_variables()
        print('raw users:{} raw locations:{}'.format(
            len(data_generator.data), len(data_generator.venues)))
        print('final users:{} final locations:{}'.format(
            len(data_generator.data_neural), len(data_generator.vid_list)))
    elif args.command == 'dir':
        process_directory(args.in_dir, args.out_dir, args.train, args.train_split)
