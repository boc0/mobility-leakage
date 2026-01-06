import pandas as pd
import numpy as np
import datetime
import os
import pickle
from time import time
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import json

# Normalized column names definition
DEVICEID = 'DeviceID'
LATITUDE = 'Latitude'
LONGITUDE = 'Longitude'
TIMESTAMP = 'Timestamp'

LOCATION = 'Location'
WORKTIME = 'Work'
NIGHTTIME = 'Home'

def save_dict(obj, filepath):
    """Save a Python object to disk using pickle."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Create parent folders if needed
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to: {filepath}")

def load_dict(filepath):
    """Load a Python object from a pickle file."""
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded from: {filepath}")
    return obj


def read_geolife_plt(filepath):
    """Convert the content of a PLT file from the Geolife dataset into a pandas dataframe.
    A processing is done to remove useless columns.

    Args:
        filepath (string): Path to the PLT file.

    Returns:
        pandas.DataFrame: Conversion of the content of the PLT file in a DataFrame.
    """
    column_names_type = {"Latitude": float,  # Latitude in decimal degrees
                         "Longitude": float, # Longitude in decimal degrees
                         "0": int,           # All set to 0 for this dataset
                         "Altitude": float,  # Altitude in feet (-777 if not valid)
                         "Days": float,      # Number of days w/ fractional part since 30/12/1899
                         "Date": str,        # Date as a string
                         "Time": str,        # Time of the day as a string
                        }
    columns_to_remove = ["0", "Days"]
    columns_to_use = [col for col in column_names_type if col not in columns_to_remove]

    df = pd.read_csv(filepath,
                skiprows=6, names=column_names_type.keys(), dtype=column_names_type,
                usecols=columns_to_use
                )
    return df


def read_all_geolife(geolife_directory_path: Path, print_runtime = False):
    """Converts and returns all the data from the geolife dataset into a single Pandas DataFrame.

    Args:
        geolife_directory_path (pathlib.Path): Path to the 'Data' folder of the Geolife dataset.
        print_runtime (Boolean): True to print the time the function took to execute.

    Returns:
        pandas.DataFrame: Dataframe containing all the data from Geolife.
    """
    
    # The loops work as follow:
    # Iterate over all users (all folders from 'Data')
    #   Iterate over all trajectory PLT files from the user i
    #       Convert the PLT file j into a dataframe containing the trajectory j from user i
    #       Add a column 'TrajectoryID' to dataframe j
    #       Insert the dataframe j into a list containing all trajectories from user i
    #   Merge all dataframes from user i into a single dataframe i containing all trajectories from the user
    #   Add a 'UserID' column to dataframe i
    #   Insert the dataframe i into a list containing all user's trajectories
    # Merge all user's trajectories into a single dataframe
    
    begin = time()
    
    trajectories_count = 0 # Count of the total number of trajectories
    all_users_df_list = [] # List containing dataframes of each user's points

    try:
        for user_id in os.listdir(geolife_directory_path):
            dir_path_user_trajectories = geolife_directory_path / user_id / 'Trajectory'
            single_user_trajectories_list = [] # List of every trajectory of a single user
            
            for trajectory_filename in os.listdir(dir_path_user_trajectories):
                trajectory_df = read_geolife_plt(dir_path_user_trajectories / trajectory_filename)
                trajectory_df.insert(0, "TrajectoryID", trajectories_count)
                single_user_trajectories_list.append(trajectory_df)
                trajectories_count += 1
                
            single_user_all_trajectories = pd.concat(single_user_trajectories_list, axis=0)
            single_user_all_trajectories.insert(0, "UserID", int(user_id))
            all_users_df_list.append(single_user_all_trajectories)
    except Exception as e:
        print(f"UserID: {user_id}")
        print(f"TrajectoryFilename: {trajectory_filename}")
        raise(e)
        
    geolife_df = pd.concat(all_users_df_list, axis=0)
    geolife_df.reset_index(drop=True, inplace=True)
    
    end = time()
    
    if print_runtime: print("Runtime of read_all_geolife: {runtime:.2f} seconds.".format(runtime=end-begin) )
    
    return geolife_df


def import_normalized_dataset(dataset_name, convert_timestamps_to_datetime = True, print_duration = False):
    """Import the normalized dataset from its pkl file.

    Args:
        dataset_name (str): The name of the dataset to import from ['geolife', 'shenzhenurban', 'yjmob100k'].
        convert_timestamps (bool, optional): Convert the timestamp column into datetime values. Defaults to True.
        print_duration (bool, optional): Print the duration of the import. Defaults to False.
    
    Returns:
        pandas.DataFrame: The imported dataset.
    """
    
    datasets_paths = {
    'geolife':       'geolife.pkl',
    'shenzhenurban': 'shenzhenurban.pkl',
    'yjmob100k':     'yjmob100k.pkl',
    'boston':        'boston.pkl',
    'shanghaikaggle': 'shanghaikaggle.pkl'
    }
    
    dataset_path = Path(r"PreprocessedData/0NormalizedCols") / datasets_paths[dataset_name]

    # Import the dataset
    begin = time()
    if print_duration: print(f"Loading dataset from {dataset_path}\n...")
    dataset = pd.read_pickle(dataset_path, compression=None)
    end = time()
    if print_duration: print(f"Dataset loaded in {(end-begin):.2f} seconds!\n")

    # Convert TIMESTAMP into datetime objects
    if convert_timestamps_to_datetime:
        begin = time()
        if print_duration: print(f"Converting str dates into datetime objects\n...")
        dataset[TIMESTAMP] = pd.to_datetime(dataset[TIMESTAMP], format='%Y-%m-%d-%H:%M:%S')
        end = time()
        if print_duration: print(f"Converted in {(end-begin):.2f} seconds!\n")

    return dataset


def time_discretization(dataset, time_interval):
    
    """Discretize the input dataset into temporal bins of the size of time_interval.
    The record to keep from the aggregation is defined as the most present location on the time interval.

    Args:
        dataset (pandas.DataFrame): The input dataset to temporally discretize. Must be with normalized columns and a TIMESTAMP column in the datetime format.
        time_interval (int): The time interval in minutes. Recommended either 30 or 60 minutes.
        selection_method (str, optional): The method to select the record in the aggregation, to choose in ['first', 'last']. Defaults to 'first'.

    Returns:
        pandas.DataFrame: Temporally discretized dataset with only one record per temporal interval.
    """
    
    # Floor the timestamp values to the previous time_interval minutes before discretisation
    dataset = dataset.copy()
    dataset[TIMESTAMP] = dataset[TIMESTAMP].dt.floor(f'{time_interval}min')
    
    # Aggregate to find the most frequent (LATITUDE, LONGITUDE) for each (DEVICEID, TIMESTAMP)
    grouped = dataset.groupby([DEVICEID, TIMESTAMP])
    
    result = []
    for (device_id, timestamp), group in grouped:
        # Count combinations of (LATITUDE, LONGITUDE)
        #most_frequent = Counter(zip(group[LATITUDE], group[LONGITUDE])).most_common(1)[0][0]
        most_frequent = Counter(zip(group[LOCATION])).most_common(1)[0][0]
        result.append([device_id, timestamp, *most_frequent])

    # Create the resulting DataFrame
    dataset = pd.DataFrame(result, columns=[DEVICEID, TIMESTAMP, LOCATION])
    
    return dataset

def users_filter(dataset, time_interval, inferred = None, min_number_of_days = 5, min_percentage_of_records_per_day = 0.5, filter_not_inferred=True):
    """Filter users based on the following criteria:
    - The user must have at least min_number_of_days days of data, each with at least min_percentage_of_records_per_day % of records compared to a full day.

    Args:
        dataset (pandas.DataFrame): The dataset to filter. The input dataset must already be time discretized.
        time_interval (int): The interval of discretization used on the input dataset.
        inferred (pandas.DataFrame, optional): The dataframe with inferred home/work locations, to filter. Defaults to None.
        min_number_of_days (int, optional): Minimum number of days N that must have P% of records. Defaults to 5.
        min_percentage_of_records_per_day (float, optional): Percentage P of records to have in a day. Defaults to 0.5.
        filter_not_inferred (bool, optional): If to filter individuals missing the home or work inference. Only possible if inferred is defined. Defaults to True.

    Returns:
        pandas.DataFrame: The dataset with filtered users. If 'inferred' is included, then also returns the filtered 'inferred', else None.
    """
    
    min_nbr_records_per_day = int(min_percentage_of_records_per_day * 1440 / time_interval)
    
    # Remove rows with undefined (NaN) location to later compute the number of records
    df = dataset.copy()
    df.dropna(subset=[LOCATION], axis='index', how='any', inplace=True)
    
    # Only keep the dates from the TIMESTAMP column
    df[TIMESTAMP] = df[TIMESTAMP].dt.date
    
    # Count of the number of records per user per day
    df = df.groupby(by=[DEVICEID, TIMESTAMP], as_index=False).size()
    
    # Determine the top N days with the most records
    #df = df.sort_values([DEVICEID, 'size'], ascending=[True, False]).groupby(DEVICEID).head(min_number_of_days)
    
    # Determine users with at least N days of minimum P records
    df = df[df['size'] >= min_nbr_records_per_day].groupby(by=DEVICEID, as_index=False).size() # Number of days with at least P% records
    eligible_users = df[df['size'] >= min_number_of_days][DEVICEID]
    
    user_filtered_inferred = None
    if inferred is not None:
        missing_inference_users = inferred[inferred.isna().any(axis=1)].index # Users missing a home or work inference
        eligible_users = [user for user in eligible_users if user not in missing_inference_users]
        user_filtered_inferred = inferred[inferred.index.isin(eligible_users)]
    
    # Filtered dataset
    user_filtered_dataset = dataset[dataset[DEVICEID].isin(eligible_users)]
    
    return user_filtered_dataset, user_filtered_inferred


def spatial_filter(dataset: 'pandas.DataFrame'):
    """Filter out records from GeoLife that are outside of Beijing.

    Args:
        dataset (pandas.DataFrame): The input dataset to filter. Must be with normalized columns.

    Returns:
        pandas.DataFrame: Spatially filtered dataset.
    """
    MIN_LATITUDE, MAX_LATITUDE = 36.0, 43.0
    MIN_LONGITUDE, MAX_LONGITUDE = 114.0, 121.0
    
    spatial_filter = (dataset[LATITUDE] > MIN_LATITUDE) & (dataset[LATITUDE] < MAX_LATITUDE) & (dataset[LONGITUDE] > MIN_LONGITUDE) & (dataset[LONGITUDE] < MAX_LONGITUDE)
    dataset_filtered = dataset[spatial_filter]
    
    return dataset_filtered


# In the RNN paper, the census tracts are used as areas
def spatial_discretization(dataset, tile_dimensions, already_discretized=False):
    """Discretize the input dataset into a grid.
    The argument tile_dimensions define the approximate dimensions for each tile.
    
    Args:
        dataset (pandas.DataFrame): Input dataset to discretize.
        tile_dimensions (tuple): Tuple (tile_size_latitude, tile_size_longitude) with the dimensions of a unit of the grid in meters.
        already_discretized (bool, optional): If the dataset is already discretized (ShenzhenUrban & YJMob100K), a particular labelization is performed. Defaults to False.
        
    Returns:
        pandas.DataFrame: Spatially discretized dataset.
        dict: Dictionary of label: coordinates.
        list: [Number of tiles (latitude axis), Number of tiles (longitude axis), Dimension of the latitude axis of a tile (in meters), Dimension of the longitude axis of a tile (in meter)]
    """
    
    LATITUDE_DISTANCE = 778364 # in meters
    LONGITUDE_DISTANCE = (629575 - 569095)/2 + 569095 # in meters
    
    df = dataset.copy()
    
    if already_discretized:
        
        # The ShenzhenUrban and YJMob100K datasets are already discretized
        print("Discretization into bins...")
        labels, bins_locations = pd.factorize(list(zip(df[LATITUDE], df[LONGITUDE])))
        df[LOCATION] = labels + 1 # The value 0 is for NaN
        print("Discretized!")
        
        print("Generation of the label:location dictionary...")
        bins_to_locations = {
            row[LOCATION]: (bins_locations[row[LOCATION] - 1]) for _, row in df.iterrows()
        }
        bins_to_locations = pd.DataFrame.from_dict(bins_to_locations, columns=[LATITUDE, LONGITUDE], orient='index')
        bins_to_locations.rename_axis(LOCATION, inplace=True)
        print("Dictionary generated!")
        
        df.drop(labels=[LATITUDE, LONGITUDE], axis='columns', inplace=True)
        
        return df, bins_to_locations, [None, None, None, None]
    
    # Compute the number of bins necessary for each dimension
    nbr_bins_lat = int(LATITUDE_DISTANCE / tile_dimensions[0])
    nbr_bins_lon = int(LONGITUDE_DISTANCE / tile_dimensions[1])
    
    # Dimensions of a tile
    tile_size_lat = LATITUDE_DISTANCE / nbr_bins_lat
    tile_size_lon = LONGITUDE_DISTANCE / nbr_bins_lon
    
    # Discretize each dimension according to the number_bins value
    print("Discretization into bins...")
    df[LATITUDE], bins_limits_latitude = pd.cut(dataset[LATITUDE], retbins=True,
                                                    labels=list(range(nbr_bins_lat)), bins=nbr_bins_lat)
    df[LATITUDE] = df[LATITUDE].astype('uint32')
    df[LONGITUDE], bins_limits_longitude = pd.cut(dataset[LONGITUDE], retbins=True,
                                                    labels=list(range(nbr_bins_lon)), bins=nbr_bins_lon)
    df[LONGITUDE] = df[LONGITUDE].astype('uint32')
    print("Discretized!")
    
    # Compute the location box number
    print("Computation of the labels...")
    df[LOCATION] = nbr_bins_lon * df[LATITUDE] + df[LONGITUDE] + 1
    print("Labels computed!")
    
    # Create a dictionary label: coordinates
    print("Generation of the label:location dictionary...")
    bins_latitudes = (bins_limits_latitude[df[LATITUDE]] + bins_limits_latitude[df[LATITUDE] + 1]) / 2
    bins_longitudes = (bins_limits_longitude[df[LONGITUDE]] + bins_limits_longitude[df[LONGITUDE] + 1]) / 2
    bins_to_locations = dict(zip(df[LOCATION], zip(bins_latitudes, bins_longitudes)))
    bins_to_locations = pd.DataFrame.from_dict(bins_to_locations, columns=[LATITUDE, LONGITUDE], orient='index')
    bins_to_locations.rename_axis(LOCATION, inplace=True)
    print("Dictionary generated!")
    
    df.drop(labels=[LATITUDE, LONGITUDE], axis='columns', inplace=True)
    
    return df, bins_to_locations, [nbr_bins_lat, nbr_bins_lon, tile_size_lat, tile_size_lon]


def fill_missing_timestamps(dataset, time_interval, chunk_size=250000):
    """Fill missing timestamps of day in the dataset with NaN records.
    This function returns the input dataset with only complete days, filled using NaN values.
    **NOTE:** The dataset must have been time discretized prior to using this function.

    Args:
        dataset (pandas.DataFrame): The (time discretized) input dataset to fill.
        time_interval (int): The time discretization interval in minutes.
        chunk_size (int, optional): The size of chunks of data for the composite processing. Default to 250000.

    Returns:
        pandas.DataFrame: Dataset filled with NaN values.
    """
    # Ensure TIMESTAMP is in datetime format
    dataset = dataset.copy()
    #dataset[TIMESTAMP] = pd.to_datetime(dataset[TIMESTAMP])
    
    # Extract unique DEVICEID-Date combinations
    dataset['Date'] = dataset[TIMESTAMP].dt.date
    unique_dates = dataset[[DEVICEID, 'Date']].drop_duplicates()
    
    # Drop intermediate 'Date' column
    dataset.drop(columns='Date', inplace=True)
    
    merged_chunks = []

    print("Processing in chunks...")
    for i in tqdm(range(0, len(unique_dates), chunk_size)):
        chunk = unique_dates.iloc[i:i + chunk_size]
        
        # Generate the full ranges in a vectorized manner for the current chunk
        # Expand each unique DEVICEID-Date into full timestamp ranges
        date_ranges = chunk['Date'].map(
            lambda d: pd.date_range(
                start=pd.Timestamp(d).replace(hour=0, minute=0, second=0),
                end=pd.Timestamp(d).replace(hour=23, minute=59, second=59),
                freq=f'{time_interval}T'
            )
        )

        # Use broadcasting to replicate DEVICEID and timestamps efficiently
        all_timestamps_chunk = pd.DataFrame({
            DEVICEID: np.repeat(chunk[DEVICEID].values, date_ranges.map(len).values),
            TIMESTAMP: np.concatenate(date_ranges.values)
        })

        # Merge with the original dataset to fill missing timestamps
        merged_chunk = pd.merge(all_timestamps_chunk, dataset, on=[DEVICEID, TIMESTAMP], how='left')
        merged_chunks.append(merged_chunk)
    
    # Concatenate all chunks
    print("Concatenating all chunks...")
    merged_df = pd.concat(merged_chunks, ignore_index=True)
    print("Timestamps filled!")

    return merged_df

def label_home_work_hours(dataset: 'pandas.DataFrame', begin_work_hour: int, end_work_hour: int, begin_night_hour: int, end_night_hour: int, begin_workpause_hour:int=None, end_workpause_hour:int=None):
    """Label records based on if the time corresponds to *work*, *home* or else.
    Returns the input dataset with two new columns WORKTIME and NIGHTTIME with boolean values.
    **NOTE:** The time is in the 24h system.

    Args:
        dataset (pandas.DataFrame): Input dataset with records to label.
        begin_work_hour (int): The hour to begin work.
        end_work_hour (int): The hour to end work.
        begin_night_hour (int): The hour to begin night.
        end_night_hour (int): The hour to end night.
        begin_workpause_hour (int, optional): The hour to begin the lunch pause at work if there is one. Defaults to None.
        end_workpause_hour (int, optional): The hour to end the lunch pause at work if there is one. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    dataset = dataset.copy()
    
    # Work inference
    begin_work = datetime.time(hour=begin_work_hour, minute=0, second=0) # included
    end_work = datetime.time(hour=end_work_hour, minute=0, second=0) # not included
    
    if (begin_workpause_hour is not None) and (end_workpause_hour is not None):
        begin_pause = datetime.time(hour=begin_workpause_hour, minute=0, second=0) # not included
        end_pause = datetime.time(hour=end_workpause_hour, minute=0, second=0) # included
    else:
        begin_pause = begin_work
        end_pause = begin_work
    
    timestamp_times = dataset[TIMESTAMP].dt.time

    mask_morning = (begin_work <= timestamp_times) & (timestamp_times < begin_pause)
    mask_afternoon = (end_pause <= timestamp_times) & (timestamp_times < end_work)

    dataset[WORKTIME] = mask_morning | mask_afternoon
    
    # Night inference
    begin_night = datetime.time(hour=begin_night_hour, minute=0, second=0) # included
    end_night = datetime.time(hour=end_night_hour, minute=0, second=0) # not included

    midnight_m = datetime.time(hour=23, minute=59, second=59) # included
    midnight_p = datetime.time(hour=0, minute=0, second=0) # included
    
    mask_beforemidnight = (begin_night <= timestamp_times) & (timestamp_times <= midnight_m)
    mask_aftermidnight = (midnight_p <= timestamp_times) & (timestamp_times < end_night)

    dataset[NIGHTTIME] = mask_beforemidnight | mask_aftermidnight
    
    return dataset

def infer_most_present_locations(dataset, time_to_infer: str):
    """Infer the most present locations at a certain timespan.

    Args:
        dataset (pandas.DataFrame): The input dataset for which to infer the most present locations.
        time_to_infer (str): The timespan to consider. Must be in [WORKTIME, NIGHTTIME].
        
    Returns:
        pandas.DataFrame: A dataframe with two columns DEVICEID and LOCATION, the latter value corresponding to the most present location of the first.
    """
    # Initialize the dictionary
    inferred_locations = {did: None for did in dataset[DEVICEID].unique()}
    def aggregation_function(col):
        # Function to infer the most present location
        # If the input column contains only NaN values then return NaN
        col_mode = col.mode()
        return col_mode[0]
    # What to do when there are multiple most present locations? *Random*? Or maybe just remove users if inference is impossible?
    # The most present location is selected at random (first value in list)
    df_at_timespan = dataset[dataset[time_to_infer]][[DEVICEID, LOCATION]]
    most_present_locations = df_at_timespan.dropna(axis='index').groupby(by=DEVICEID, as_index=False).agg(aggregation_function)
    full_inference_dict = {
        row[DEVICEID]: int(row[LOCATION])
        for index, row in most_present_locations.iterrows()
    }
    inferred_locations.update(full_inference_dict)
    return inferred_locations



def fill_infered_NaN_values(dataset):
    """Complete NaN values for both WORKTIME and NIGHTTIME timespans by infering home and work locations.

    Args:
        dataset (pandas.DataFrame): The input dataset on which to fill the NaN values.

    Returns:
        (pandas.DataFrame, dict): The filled dataset and the inferred locations for both timespans.
    """
    
    # Copy of the dataset
    df = dataset.copy()
    
    inferred_locations = dict()
    
    for timespan in [WORKTIME, NIGHTTIME]:
    
        # Dictionary containing the most present location at timespan for each DEVICEID
        print(f"Creating dictionary of most present location at {timespan} for each DEVICEID...")
        begin = time()
        location_filling_dictionary = infer_most_present_locations(df, timespan)
        end = time()
        print(f"Dictionary created in {end-begin:.2f}s!")
        
        # Create a mask for rows where LOCATION is NaN and timespan is True
        print(f"Creating the mask for NaN values filling at {timespan}...")
        begin = time()
        timespan_mask = (df[timespan]) & (df[LOCATION].isna())
        end = time()
        print(f"Mask created in {end-begin:.2f}s!")
        
        # Use the filling dictionary to assign values for the masked rows
        print(f"Assigning the values through mapping for {timespan}...")
        begin = time()
        df.loc[timespan_mask, LOCATION] = df.loc[timespan_mask, DEVICEID].map(location_filling_dictionary)
        end = time()
        print(f"Values assigned in {end-begin:.2f}s!\n")
        
        inferred_locations[timespan] = location_filling_dictionary
        
        df.drop(columns=timespan, inplace=True)
        
    inferred_locations = pd.DataFrame.from_dict(inferred_locations, orient='columns').reset_index().rename(columns={'index': DEVICEID})
    inferred_locations.set_index(DEVICEID, inplace=True)

    return df, inferred_locations


def extract_best_days(dataset, dictionary, number_of_best_days):
    """Extraction of the N best days for each individual from the dataset.

    Args:
        dataset (pandas.DataFrame): Input dataset.
        dictionary (pandas.DataFrame): Input mapping of label:coordinates.
        number_of_best_days (int): Number of best days to keep.

    Returns:
        (pandas.DataFrame, pandas.DataFrame): Returns (dataset, dictionary) with the N best days only for each individual.
    """
    
    # Best N days for each individual
    best_days = (dataset
        .assign(Date=lambda df: df[TIMESTAMP].dt.date) # Convert timestamps into only dates
        .dropna(axis=0, how='any') # Remove NaN
        .groupby(by=[DEVICEID, 'Date'], as_index=False).size() # Get the number of records per day
        .sort_values(by=[DEVICEID, 'size'], ascending=[True, False]) # Sort the number of records/day
        .groupby(by=DEVICEID).head(number_of_best_days) # Extract the N best days
        .drop(columns='size')
    )
    
    # Dataset with only the best N days
    dataset_extracted = (dataset
        .assign(Date=lambda df: df[TIMESTAMP].dt.date)
        .merge(best_days, on=[DEVICEID, 'Date'], how='inner')
        .drop(columns='Date')
    ) # Intersection of the dataset and best_days on DEVICEID and the date
    
    # Dictionary with only the best N days
    dictionary_extracted = (dictionary
        .merge(dataset_extracted[LOCATION], on=[LOCATION], how='inner')
        .drop_duplicates(subset=LOCATION)
        .set_index(LOCATION)
    )
    
    return dataset_extracted, dictionary_extracted


def relabelize(dataset, inferred, dictionary):
    """Relabelization to lower integers.

    Args:
        dataset (pandas.DataFrame): Input dataset.
        inferred (pandas.DataFrame): Input inferred home/work labels dictionary.
        dictionary (pandas.DataFrame): Input mapping of label:coordinates.

    Returns:
        (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame): Returns (dataset, inferred, dictionary) with relabelled location labels.
    """
    
    # Mapper for the relabelization into smaller integers
    relabelization = {old_label: new_label[0] for new_label, old_label in np.ndenumerate(np.concat([[np.nan], np.sort(dictionary.index)]))}
    
    # Relabeled inferred home/work labels
    # Infered locations not in the dataset (because remove when selecting N best days are filled with 0)
    inferred_relabeled = inferred.copy()
    inferred_relabeled[WORKTIME] = inferred_relabeled[WORKTIME].map(relabelization).fillna(0).astype('uint64')
    inferred_relabeled[NIGHTTIME] = inferred_relabeled[NIGHTTIME].map(relabelization).fillna(0).astype('uint64')
    
    # Relabeled dictionary
    dictionary_relabeled = dictionary.rename(mapper=relabelization)
    
    # Relabeled dataset
    dataset_relabeled = dataset.copy()
    dataset_relabeled[LOCATION] = dataset_relabeled[LOCATION].map(relabelization)

    return dataset_relabeled, inferred_relabeled, dictionary_relabeled


def generate_strings(dataset, inferred):
    """Generate the two input for the generative model.

    Args:
        dataset (pandas.DataFrame): Input dataset.
        inferred (pandas.DataFrame): Input inferred home/work labels.

    Returns:
        (str, dict): Returns the string of trajectories and the dictionary of prefixes to counts.
    """
    
    txt = ''
    prefixes_to_counts = {}

    for id, group in dataset.sort_values(by=[DEVICEID, TIMESTAMP], ascending=[True, True]).groupby(by=DEVICEID, as_index=False):
        home = inferred.loc[id][NIGHTTIME]
        work = inferred.loc[id][WORKTIME]
        prefix = f"{home} {work}"
        if prefix in prefixes_to_counts:
            prefixes_to_counts[prefix] += 1
        else:
            prefixes_to_counts[prefix] = 1
        txt += f"{prefix} {' '.join([str(loc) for loc in group[LOCATION]])}\n"
    
    return txt, prefixes_to_counts

#### Data Splitting ####

DATASET_NAME_TO_FOLDER_NAME = {
    'boston': 'Boston',
    'geolife': 'Geolife',
    'shenzhenurban': 'ShenzhenUrban',
    'yjmob100k': 'YJMob100Kv3',
    'shanghaikaggle': 'ShanghaiKaggle'
}

def import_dataset(dataset_name:str, dataset_version=0):
    """Import the named dataset using its version.
    NOTE: If the Berke et al. dataset 'boston' is called, then dataset_version is useless.

    Args:
        dataset_name (str): Name of the dataset to import from ['boston', 'geolife', 'shenzhenurban', 'yjmob100k']
        dataset_version (int, optional): Version of the dataset to import. Default to 0.

    Returns:
        (str, str): Returns two strings: the first with the trajectories and the second with a description of the dataset.
    """
    
    if dataset_name == 'boston':
        with open(Path('../data/relabeled_trajectories_1_workweek.txt'), 'r') as f:
            trajectories = f.read()
        description = 'The dataset used in the Berke et al. paper.'
        statistics = {
            'name': "boston",
            'version': None,
            'time_interval': 60,
            'nbr_unique_locations': 652,
            'workhours': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'nighthours': [0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23]
            }
        return trajectories, description, statistics
    
    folder_path = Path('./data/') / DATASET_NAME_TO_FOLDER_NAME[dataset_name] / str(dataset_version)
    
    # Import of the trajectories and description
    with open(folder_path / 'trajectories.txt') as f:
        trajectories = f.read()
    with open(folder_path / 'parameters.txt') as f:
        description = f.read()
    with open(folder_path / 'stats.json') as f:
        statistics = json.load(f)
        
    return trajectories, description, statistics

def data_folder_path(dataset_name, dataset_version=0, create_folder=True):
    """Compute the path to the train, generation and evaluation data from the dataset name and version.
    If the folder doesn't exist and create_folder == True then it's created.
    NOTE: If the Berke et al. dataset 'boston' is called, then dataset_version is useless.

    Args:
        dataset_name (str): Name of the dataset to import from ['boston', 'geolife', 'shenzhenurban', 'yjmob100k']
        dataset_version (int, optional): Version of the dataset to import. Default to 0.
        create_folder (bool, optional): If the folder must be created. Default to True.
    Returns:
        pathlib.Path: Path to the folder.
    """
    
    # Output folder
    folder_path = Path('../LLM/data/') / DATASET_NAME_TO_FOLDER_NAME[dataset_name] / (str(dataset_version) if dataset_name != 'boston' else '')
    
    # Creation of the output folder if it doesn't exist
    if create_folder and (not os.path.exists(folder_path)):
        os.makedirs(folder_path)
    
    return folder_path
