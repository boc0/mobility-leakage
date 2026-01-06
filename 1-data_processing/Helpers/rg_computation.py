"""
    This library provides functions to compute three version of the radius of gyration (rg) mobility metric, as detailed in the paper [Chen et al.](https://ieeexplore.ieee.org/document/8475047). 

    Functions:
    - rg_unique.
    - rg_event.
    - rg_time.
    - harvesine_distance.
"""
import pandas as pd
import numpy as np
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two (lat, lon) locations using the harvesine formula

    Parameters:
    - lat1, lon1 (float): geographical coordinates of the first location.
    - lat2, lon2 (float): geographical coordinates of the second location.

    Returns:
    float: The distance in km.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    
    # Calculate the distance
    distance = c * r
    
    return distance

def haversine_distance_vector(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of the Earth in kilometers (average radius)
    R = 6371.0 
    
    # Calculate the distance
    distance = R * c
    return distance

def rg_unique(df):
    """
    Calculate the radius of gyration of users following the "rg_unique" algorithm

    Parameters:
    - df (pandas dataframe): with columns id, latitude, longitude, timestamp (pandas datetime)

    Returns:
    pandas dataframe: with the columns id, rg_unique (in km)
    """
    df_rg = df[['id', 'latitude', 'longitude']].drop_duplicates()
    df_rg_info = df_rg.groupby('id').size().reset_index(name='N')
    df_rg_info = df_rg_info.merge(right=df_rg.groupby('id').latitude.mean().reset_index(name='rcm_lat'), on='id')
    df_rg_info = df_rg_info.merge(right=df_rg.groupby('id').longitude.mean().reset_index(name='rcm_lon'), on='id')
    df_rg = df_rg.merge(right=df_rg_info, on='id', how='left')

    df_rg['distance'] = df_rg.apply(lambda x: haversine_distance(x['latitude'], x['longitude'], x['rcm_lat'], x['rcm_lon']), axis=1)
    df_rg['tmp'] = df_rg.distance**2/df_rg.N
    df_output = np.sqrt(df_rg.groupby('id').tmp.sum()).reset_index(name='rg_unique')

    return df_output

def rg_event(df):
    """
    Calculate the radius of gyration of users following the "rg_event" algorithm

    Parameters:
    - df (pandas dataframe): with columns id, latitude, longitude, timestamp (pandas datetime)

    Returns:
    pandas dataframe: with the columns id, rg_event (in km)
    """
    df_rg = df.groupby(['id', 'latitude', 'longitude']).size().reset_index(name='mk') 
    df_rg_info = df_rg.groupby('id').apply(lambda x: (x['latitude']*x['mk']).sum()/(x['mk'].sum())).reset_index(name='rcm_lat')
    df_rg_info = df_rg_info.merge(
        right = df_rg.groupby('id').apply(lambda x: (x['longitude']*x['mk']).sum()/(x['mk'].sum())).reset_index(name='rcm_lon'), 
        on='id')
    df_rg = df_rg.merge(right= df_rg_info, on='id', how='left')
    df_rg['distance'] = df_rg.apply(lambda x: haversine_distance(x['latitude'], x['longitude'], x['rcm_lat'], x['rcm_lon']), axis=1)
    df_output = df_rg.groupby('id').apply(lambda x: np.sqrt(((x['distance']**2)*x['mk'].sum())/(x['mk'].sum())) ).reset_index(name='rg_event')
    return df_output[['id', 'rg_event']]

def rg_time(df, dur=30):
    """
    Calculate the radius of gyration of users following the "rg_time" algorithm

    Parameters:
    - df (pandas dataframe): with columns id, latitude, longitude, timestamp (pandas datetime)

    Returns:
    pandas dataframe: with the columns id, rg_time (in km)
    """
    df['rounded_timestamp'] = pd.to_datetime(df['timestamp'].dt.floor(f'{dur}T'))
    df_rg = df.groupby(['id', 'latitude', 'longitude']).rounded_timestamp.nunique().reset_index(name='sk')
    df_rg_info = df_rg.groupby('id').apply(lambda x: (x['latitude']*x['sk']).sum()/(x['sk'].sum())).reset_index(name='rcm_lat')
    df_rg_info = df_rg_info.merge(
        right = df_rg.groupby('id').apply(lambda x: (x['longitude']*x['sk']).sum()/(x['sk'].sum())).reset_index(name='rcm_lon'), 
        on='id')
    df_rg = df_rg.merge(right= df_rg_info, on='id', how='left')
    df_rg['distance'] = df_rg.apply(lambda x: haversine_distance(x['latitude'], x['longitude'], x['rcm_lat'], x['rcm_lon']), axis=1)
    df_output = df_rg.groupby('id').apply(lambda x: np.sqrt(((x['distance']**2)*x['sk'].sum())/(x['sk'].sum())) ).reset_index(name='rg_time')
    return df_output[['id', 'rg_time']]
