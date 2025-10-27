import numpy as np
import time
from numpy import *
from scipy import io
import signal
import psutil
import gc
from functools import partial
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

import sys
# sys.path.append("../")

from collections import OrderedDict
from copy import copy
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
import os
import xarray as xr
import time
import glob
import multiprocessing as mp
import threading as th
import concurrent.futures as cf

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool, cpu_count
import joblib
import warnings
warnings.filterwarnings("ignore")


torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
import random
random.seed(1)

import torch

import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
from torch.autograd import Variable
from sklearn.preprocessing import PolynomialFeatures
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.io import loadmat, savemat

# from fancyimpute import KNN
cpu_count()

NUM_WORKERS = os.cpu_count() - 1
CHUNK_SIZE = 1000
print(f"Setting up parallel processing with {NUM_WORKERS} workers")


def inverse_transform_predictions(scaled_predictions, scalers):
    """
    Convert scaled predictions back to original scale
    Args:
        scaled_predictions: scaled predictions array
        scalers: dictionary of fitted scalers per feature
    Returns:
        original_scale_predictions 
    """
    feature_names = ['WSPD10', 'WDIR10', 'Obs_SO2', 'CO', 'Obs_O3', 'Obs_NO2', 'Obs_PM10', 'PM25']
    original_scale = np.zeros_like(scaled_predictions)
    
    for i, feature in enumerate(feature_names):
        original_scale[..., i] = scalers[feature].inverse_transform(
            scaled_predictions[..., i].reshape(-1, 1)).reshape(scaled_predictions[..., i].shape)
            
    return original_scale


nodes = OrderedDict()
folder_path = 'station_wise/'
with open('stations_info_183_lat_lon.csv', 'r') as f:# _201912_distance.csv', 'r')
    for line in f:
        index_long, index_lat, index  = line.rstrip('\n').split(',') # index, longitude, latitude, index_long, index_lat, dist_grid = line.rstrip('\n').split(',')
        index = int(index)
        file_name = 'staion_' + str(index) + '.nc'
        file_path = str(folder_path + file_name)
        
        assert os.path.isfile(file_path)
        latitude = xr.open_dataset(file_path).lat
        longitude = xr.open_dataset(file_path).lon
        longitude, latitude = float(longitude), float(latitude)
        nodes.update({index: {'lon': longitude, 'lat': latitude}})

start = time.time()
year_str = 2018
year_end = 2021
path_obs = 'Input_data/Obs_ext/'
file_path = []
start = time.time()

num_processes = 4
num_threads = 2

new_dates = pd.date_range(start='2022-01-01 00:00:00', end='2022-01-01 12:00:00', freq='h')

for station in range(0, len(list(nodes.items()))):
    for year in range(year_str, year_end + 1):
        path_og = path_obs + str(year) + '/' + str(list(nodes.items())[station][0]) 
        file_path.append(glob.glob(path_og + '*.csv'))
print('Total files are ', len(file_path))


def process_file(filename):
    obs_single = pd.read_csv(filename)
    obs_single['Unnamed: 0'] = pd.to_datetime(obs_single['Unnamed: 0'], format='%Y-%m-%d %H:%M:%S')
    obs_single.rename(columns={'Unnamed: 0': 'Date'}, inplace = True)
    # print(filename[41:45], filename[46:52], filename)
    if filename[41:45]=='2016':
        obs_single = obs_single.drop(obs_single.index[:13])
        obs_single['station'] = filename[46:52]
        return obs_single
    
    if filename[41:45]=='2021':
        new_data = pd.DataFrame(columns=obs_single.columns)
        new_data['Date'] = new_dates # pd.DataFrame(new_dates)
        new_data['Station_ID'] = obs_single['Station_ID'][:13]
        obs_single = pd.concat([obs_single, new_data], ignore_index = True)
        obs_single['station'] = filename[46:52]
        return obs_single
    else:
        obs_single['station'] = filename[46:52]
        return obs_single


def process_chunk(chunk):
    with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = [executor.submit(process_file, filename) for filename in chunk]
        cf.wait(results)
        chunk_result = [f.result() for f in results]
        chunk_result = pd.concat(chunk_result, ignore_index=True)
    return chunk_result

input_files = [str(path[0]) for path in file_path]
num_files_per_chunk = len(file_path) // num_processes
file_chunks = [input_files[i:i+num_files_per_chunk] for i in range(0, len(input_files), num_files_per_chunk)]
    
pool = mp.Pool(num_processes)

results = pool.map(process_chunk, file_chunks)

obs_data = pd.concat(results, ignore_index = True)
obs_data = obs_data.drop(['Station_ID'], axis = 1)

obs_data.station = obs_data['station'].astype(int)
column_rename_dict = {'SO2': 'Obs_SO2', 'O3': 'Obs_O3', 'NO2': 'Obs_NO2', 'PM10' : 'Obs_PM10', 'station' : 'Station_ID'}
obs_data.rename(columns = column_rename_dict, inplace = True)
print(f'Observation column are: f{obs_data.columns}')
obs_data.loc[:, obs_data.columns != 'Date'][obs_data.loc[:, obs_data.columns != 'Date'] < 0] = np.nan
print("Total time it took to compute this cell is ", time.time() - start)
print('=========================================================================================')

start = time.time()
path_mcip = 'Input_data/MCIP_ext/'
station_ID = pd.read_csv('stations_info_183_lat_lon.csv', header = None)
file_path = []
for i in range(0, len(nodes.items())):
    for year in range(year_str, year_end + 1):
        if (list(nodes.items())[i][0] == int(station_ID[2][i])):
            path_og = path_mcip + str(year) + '/'
            file_path.append(path_og + str(station_ID[0][i]) + '_' + str(station_ID[1][i]) + '.csv')
print('Total files are ', len(file_path))
mcip_data_chunk = []

def process_file(filename):
    mcip_single = pd.read_csv(filename)
    mcip_single.rename(columns={'Date Local': 'Date'}, inplace = True)
    mcip_single['Date'] = pd.to_datetime(mcip_single['Date'])
    mcip_single = mcip_single.set_index('Date').asfreq('1h')
    return mcip_single


def process_chunk(chunk):
    with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = [executor.submit(process_file, filename) for filename in chunk]
        cf.wait(results)
        chunk_result = [f.result() for f in results]
        chunk_result = pd.concat(chunk_result, ignore_index=True)
    return chunk_result


input_files = [str(path) for path in file_path]
num_files_per_chunk = len(file_path) // num_processes
file_chunks = [input_files[i:i+num_files_per_chunk] for i in range(0, len(input_files), num_files_per_chunk)]

pool = mp.Pool(num_processes)

results = pool.map(process_chunk, file_chunks)
 
mcip_data = pd.concat(results, ignore_index = True)
mcip_data.rename(columns={'Date UTC': 'Date'}, inplace = True)
mcip_data = mcip_data.drop('Date', axis = 1)
print(f'MCIP column are: f{mcip_data.columns}')
print("Total time it took to compute this cell is ", time.time() - start)
print('=========================================================================================')


start = time.time()
path_cmaq = 'Input_data/CMAQ_ext/'
station_ID = pd.read_csv('stations_info_183_lat_lon.csv', header = None)
file_path = []
for i in range(0, len(nodes.items())):
    for year in range(year_str, year_end + 1):
        if (list(nodes.items())[i][0] == station_ID[2][i]):
            path_og = path_cmaq + str(year) + '/'
            file_path.append(path_og + str(station_ID[0][i]) + '_' + str(station_ID[1][i]) + '.csv')
print('Total files are ', len(file_path)) 

cmaq_data_chunk = []

def process_file(filename):
    cmaq_single = pd.read_csv(filename)
    cmaq_single.rename(columns={'Date Local': 'Date'}, inplace = True)
    cmaq_single['Date'] = pd.to_datetime(cmaq_single['Date'])
    cmaq_single = cmaq_single.set_index('Date').asfreq('1h')
    return cmaq_single


def process_chunk(chunk):
    with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = [executor.submit(process_file, filename) for filename in chunk]
        cf.wait(results)
        chunk_result = [f.result() for f in results]
        chunk_result = pd.concat(chunk_result, ignore_index=True)
    return chunk_result

input_files = [str(path) for path in file_path]
num_files_per_chunk = len(file_path) // num_processes
file_chunks = [input_files[i:i+num_files_per_chunk] for i in range(0, len(input_files), num_files_per_chunk)]
    
pool = mp.Pool(num_processes)

results = pool.map(process_chunk, file_chunks)
cmaq_data = pd.concat(results, ignore_index = True)
cmaq_data2 = cmaq_data.drop(['SO2'], axis = 1)  # 'Date UTC'
print(f'CMAQ column are: f{cmaq_data.columns}')
print("Total time it took to compute this cell is ", time.time() - start)

print(f"MCIP Shape is {mcip_data.shape} and columns are {mcip_data.columns}")
print(f"CMAQ Shape is {cmaq_data2.shape} and columns are {cmaq_data2.columns}")

start = time.time()
df_concatenated = pd.concat([mcip_data, cmaq_data2, obs_data], axis = 1)
df_concatenated = df_concatenated[~((df_concatenated.Date.dt.month==2) & (df_concatenated.Date.dt.day==29))]
print(len(df_concatenated.columns))
print(df_concatenated.columns)
print("Concatenate time is ", time.time() - start)

def add_t(arr, seq_len):
    t_len = arr.shape[0]
    assert t_len > seq_len
    arr_ts = []
    for i in range(seq_len, t_len):
        arr_t = arr[i-seq_len:i]
        arr_ts.append(arr_t)
    arr_ts = np.stack(arr_ts, axis=0)
    return arr_ts


def reshaping(x, num_features, pm):
    
    # preparing the input features in the format (number_of_days/batch_size/number_of_samples, number_of_stations, number_of features)
    # Get the number of days, number of stations, and number of features
    
    num_stations = 183
    num_days = int(len(x) / (num_stations))
    if pm:
        num_features = 1
    else:
        num_features = num_features
    
    reshaped_array = x.reshape(num_days, num_stations, num_features, order = 'F')
    # reshaped_array = add_t(reshaped_array, 24 + 72)
    
    ###########################################################################################################################
    
    return reshaped_array


dates = df_concatenated.Date.values
stations = df_concatenated.Station_ID.values
cols = ['Date', 'Station_ID', 'Date UTC', 'HFX', 'RADYNI', 'RSTOMI', 'TEMPG', 'RGRND', 
        'CFRAC', 'CLDT', 'CLDB', 'WBAR', 'SNOCOV', 'VEG', 'LAI', 'WR', 'SOIM1', 'SOIM2', 'SOIT1',
       'SOIT2', 'SLTYP', 'ISOPRENE', 'OLES', 'AROS', 'ALKS', 'USTAR', 'WSTAR', 'PM10', 'PM2P5', 'O3', 
       'NO', 'NOX', 'NO2', 'PRSFC', 'MOLI', 'GLW', 'GSW'] # 'WSPD10', 'WDIR10',, 'PBL', 'TEMP2', 'Q2', 'RN', 'RC' 

def process_wind_direction(wind_dir):
    """
    Convert wind direction to sine and cosine components
    Args:
        wind_dir: Wind direction in degrees (0-360)
    Returns:
        tuple: (wind_sin, wind_cos)
    """
    # Convert to radians
    wind_rad = np.radians(wind_dir)
    
    # Calculate components
    wind_sin = np.sin(wind_rad)
    wind_cos = np.cos(wind_rad)
    
    return wind_sin, wind_cos

def process_wind_speed(wind_speed, wind_direction):
    """
    Convert wind speed to U and V components
    Args:
        wind_speed: Wind speed in m/s
        wind_direction: Wind direction in degrees (0-360)
    Returns:
        tuple: (wind_u, wind_v)
    """
    # Convert to radians
    wind_rad = np.radians(wind_direction)
    
    # Calculate components
    wind_u = wind_speed * np.sin(wind_rad)
    wind_v = wind_speed * np.cos(wind_rad)
    
    return wind_u, wind_v

def process_cyclic_feature(values, period):
    """
    Convert cyclical features into sine and cosine components.
    
    Args:
        values: Array-like of numerical values to be transformed
        period: The period of the feature (e.g., 24 for hours, 12 for months)
    
    Returns:
        tuple: (sin_component, cos_component)
    """
    radians = 2 * np.pi * values / period
    sin_component = np.sin(radians)
    cos_component = np.cos(radians)
    return sin_component, cos_component

def add_temporal_features(df, date_column):
    """
    Add temporal features capturing daily, weekly, and seasonal patterns using an explicit date column.
    
    Args:
        df: DataFrame containing the data
        date_column: pandas Series containing datetime values
        
    Returns:
        DataFrame with added temporal features
    """
    # Create a copy to avoid modifying the original
    df_enhanced = df.copy()
    
    # Convert date column to datetime if it isn't already
    dates = pd.to_datetime(date_column)
    
    # Extract temporal components
    hours = dates.hour
    months = dates.month
    weekdays = dates.dayofweek  # Monday = 0, Sunday = 6
    dayofyear = dates.dayofyear
    
    # Process each temporal component
    hour_sin, hour_cos = process_cyclic_feature(hours, 24)
    month_sin, month_cos = process_cyclic_feature(months, 12)
    weekday_sin, weekday_cos = process_cyclic_feature(weekdays, 7)
    dayofyear_sin, dayofyear_cos = process_cyclic_feature(dayofyear, 366)
    
    # Add the temporal features to the DataFrame
    temporal_features = {
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'weekday_sin': weekday_sin,
        'weekday_cos': weekday_cos,
        'dayofyear_sin': dayofyear_sin,
        'dayofyear_cos': dayofyear_cos
    }
    
    for name, feature in temporal_features.items():
        df_enhanced[name] = feature
        
    return df_enhanced

def process_all_wind_features(wind_speed, wind_direction):
    """
    Calculate all wind-related features: U, V components and directional encodings.
    
    Args:
        wind_speed: Wind speed values in m/s
        wind_direction: Wind direction in degrees (meteorological convention)
    
    Returns:
        dict: Dictionary containing all wind features:
            - u_component: Zonal wind component
            - v_component: Meridional wind component
            - wind_dir_sin: Sine of wind direction
            - wind_dir_cos: Cosine of wind direction
            - wind_speed: Original wind speed (preserved)
    """
    # Calculate U and V components
    wind_dir_rad = np.radians(270 - wind_direction)
    u_component = -wind_speed * np.cos(wind_dir_rad)
    v_component = -wind_speed * np.sin(wind_dir_rad)
    
    # Calculate directional encodings
    wind_dir_rad_raw = np.radians(wind_direction)  # Raw direction for sin/cos
    wind_dir_sin = np.sin(wind_dir_rad_raw)
    wind_dir_cos = np.cos(wind_dir_rad_raw)
    
    return {
        'U_WIND': u_component,
        'V_WIND': v_component,
        'WSPD10': wind_speed,  # Preserve original wind speed
        'WDIR10_cos': wind_dir_cos,
        'WDIR10_sin': wind_dir_sin,
        'WDIR10': wind_direction,
    }

def positional_encoding(geo_coords, L):
    """
    Compute the positional sinusoidal encoding for latitude and longitude.
    Parameters:
    - geo_coords (numpy.ndarray): Array containing latitude and longitude in degrees.
    - L (int): Number of frequency bands.
    Returns:
    - numpy.ndarray: Positional encoding vector of size 4L.
    """
    # Convert latitude and longitude to radians
    lat = geo_coords[:, 0]
    lon = geo_coords[:, 1]
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    pe_lat = []
    pe_lon = []
    
    # Compute sinusoidal functions for each frequency band
    for i in range(L):
        frequency = 2**i * np.pi  # Fixed the formula here (removed space)
        pe_lat.append(np.sin(frequency * lat_rad))
        pe_lat.append(np.cos(frequency * lat_rad))
        pe_lon.append(np.sin(frequency * lon_rad))
        pe_lon.append(np.cos(frequency * lon_rad))
    
    # Concatenate latitude and longitude encodings
    pe_lat = np.array(pe_lat).T  # Transpose to get shape (n_samples, n_features)
    pe_lon = np.array(pe_lon).T
    pe = np.hstack([pe_lat, pe_lon])
    
    return pe

def get_station_coordinates(stations, nodes):
    """
    Map station IDs to their latitude and longitude coordinates.
    
    Args:
        stations: Array of station IDs
        nodes: OrderedDict containing station information with lat/lon coordinates
    
    Returns:
        numpy.ndarray: Array of shape (n_samples, 2) with latitude and longitude
    """
    # Initialize array to store coordinates
    coordinates = np.zeros((len(stations), 2))
    
    # Fill in coordinates for each station
    for i, station_id in enumerate(stations):
        if station_id in nodes:
            coordinates[i, 0] = nodes[station_id]['lat']
            coordinates[i, 1] = nodes[station_id]['lon']
        else:
            # Handle missing stations - could use mean or another strategy
            print(f"Warning: Station ID {station_id} not found in nodes dictionary")
            # For now, set to 0 (you may want to handle this differently)
            coordinates[i, 0] = 0
            coordinates[i, 1] = 0
    
    return coordinates


df = df_concatenated.drop(cols, axis = 1)

wind_features = process_all_wind_features(
    df['WSPD10'], 
    df['WDIR10']
)

columns_to_drop = cols + ['WSPD10', 'WDIR10']
df = df_concatenated.drop(columns_to_drop, axis = 1)

final_data = pd.DataFrame()

# Add wind features in desired order
for col in ['U_WIND', 'V_WIND', 'WSPD10', 'WDIR10_cos', 'WDIR10_sin', 'WDIR10']:
    final_data[col] = wind_features[col]

# Add temporal features
temporal_features = add_temporal_features(df, dates)
temporal_features_names = [
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'weekday_sin', 'weekday_cos',
    'dayofyear_sin', 'dayofyear_cos'
]
for col in temporal_features_names:
    final_data[col] = temporal_features[col]

geo_coords = get_station_coordinates(stations, nodes)
L = 4
pos_encodings = positional_encoding(geo_coords, L)
# print(pos_encodings.shape)

# Add positional encoding features to final_data
pos_encoding_cols = []
for i in range(pos_encodings.shape[1]):
    col_name = f'pe_{i}'
    final_data[col_name] = pos_encodings[:, i]
    pos_encoding_cols.append(col_name)

meteo_features = ['PBL', 'TEMP2', 'Q2', 'RN', 'RC']
for col in meteo_features:
    final_data[col] = df[col]

# Add pollutants in specified order
pollutant_features = ['Obs_SO2', 'CO', 'Obs_O3', 'Obs_NO2', 'Obs_PM10', 'PM25']
for col in pollutant_features:
    final_data[col] = df[col]

print("Features in final_data:", final_data.columns.tolist())
# Update feature names list to reflect new ordering
feature_names = final_data.columns.tolist()


print("\nReshaping unscaled data...")
num_features = final_data.shape[-1]
reshaped_data = np.squeeze(reshaping(final_data.values, num_features, pm = False))
print(f"Reshaped data shape: {reshaped_data.shape}")


# Initialize an array to store the NaN percentages
samples, stations, features = reshaped_data.shape
nan_percentages = np.zeros((stations, features))

# Calculate the NaN percentages for each feature at each station
for i in range(stations):
    for j in range(features):
        nan_percentages[i, j] = np.isnan(reshaped_data[:, i, j]).mean() * 100

# Convert to DataFrame for easier plotting
nan_percentages_df = pd.DataFrame(nan_percentages, columns = feature_names)

# Remove stations with NaN% greater than 15% for any variable
stations_to_keep = nan_percentages_df.max(axis=1) <= 15
filtered_data = reshaped_data[:, stations_to_keep.values, :]
nan_position = np.isnan(filtered_data)

nan_percentages_df = nan_percentages_df[stations_to_keep.values]

# Filter nodes based on stations to keep
filtered_nodes = {key: nodes[key] for key, value in zip(nodes.keys(), stations_to_keep.values) if value}
lats = np.array([node['lat'] for node in filtered_nodes.values()])
lons = np.array([node['lon'] for node in filtered_nodes.values()])
# Calculate distance matrix in kilometers
distance_threshold_km = 50  # Distance threshold in kilometers
distance_matrix = np.zeros((len(lats), len(lats)))
from geopy.distance import geodesic
for i in range(len(lats)):
    for j in range(len(lats)):
        if i != j:
            distance_matrix[i, j] = geodesic((lats[i], lons[i]), (lats[j], lons[j])).kilometers
            #np.sqrt((lats[i] - lats[j])**2 + (lons[i] - lons[j])**2) * 111

# Filter the distance matrix to only include stations within the distance threshold
filtered_distance_matrix = np.where(distance_matrix <= distance_threshold_km, distance_matrix, np.inf)

def simplified_imputation(data, distance_matrix, max_temporal_gap=24):
    """
    Simplified two-stage imputation using spline interpolation for temporal gaps
    and KNN for spatial gaps.
    
    Args:
        data: Array of shape (samples, stations, features)
        distance_matrix: Distance matrix between stations
        max_temporal_gap: Maximum gap to interpolate temporally
    
    Returns:
        Imputed data array
    """
    samples, stations, features = data.shape
    imputed_data = data.copy()
    
    print("Starting temporal imputation...")
    # Stage 1: Temporal imputation using spline interpolation
    for station in range(stations):
        for feature in range(features):
            series = pd.Series(data[:, station, feature])
            
            # Only interpolate gaps shorter than max_temporal_gap
            mask = series.isna()
            if mask.any():
                # Use cubic spline interpolation for temporal gaps
                imputed_data[:, station, feature] = series.interpolate(
                    method='linear', 
                    limit=max_temporal_gap,
                    limit_direction='both'
                ).values
    
    print("Starting spatial KNN imputation...")
    # Stage 2: KNN spatial interpolation for remaining gaps
    for sample in range(samples):
        for feature in range(features):
            # Find stations with missing values
            missing_mask = np.isnan(imputed_data[sample, :, feature])
            
            if missing_mask.any():
                valid_stations = ~missing_mask
                
                if valid_stations.any():
                    # For each station with missing value
                    for station in np.where(missing_mask)[0]:
                        # Find k nearest stations with valid data
                        distances = distance_matrix[station]
                        valid_distances = distances[valid_stations]
                        
                        if len(valid_distances) > 0:
                            # Use inverse distance weighting with k=5 nearest neighbors
                            k = min(5, len(valid_distances))
                            nearest_indices = np.argsort(valid_distances)[:k]
                            
                            # Get values and distances for nearest neighbors
                            nearest_values = imputed_data[sample, valid_stations, feature][nearest_indices]
                            nearest_distances = valid_distances[nearest_indices]
                            
                            # Compute weights using inverse distance
                            weights = 1 / (nearest_distances + 1e-6)
                            weights = weights / weights.sum()
                            
                            # Weighted average for imputation
                            imputed_value = np.sum(nearest_values * weights)
                            imputed_data[sample, station, feature] = imputed_value
    
    # Fill any remaining NaNs with feature means
    for feature in range(features):
        feature_mean = np.nanmean(imputed_data[:, :, feature])
        nan_mask = np.isnan(imputed_data[:, :, feature])
        imputed_data[:, :, feature][nan_mask] = feature_mean
    
    return imputed_data

# Use the simplified imputation
print("Starting imputation process...")
imputed_data = simplified_imputation(
    filtered_data,
    filtered_distance_matrix,
    max_temporal_gap=24
)
np.save('imputed_data.npy', imputed_data[:-1])



print("\nVerifying imputation results:")
print(f"Shape after imputation: {imputed_data.shape}")
print(f"Number of NaNs after imputation: {np.isnan(imputed_data).sum()}")


print("\nScaling imputed data...")
scaled_data = np.zeros_like(imputed_data)
scalers = {}

# Define which features should not be scaled
no_scale_features = [
    'WDIR10_cos', 'WDIR10_sin',  # Already between -1 and 1
    'hour_sin', 'hour_cos', 
    'month_sin', 'month_cos',
    'weekday_sin', 'weekday_cos',
    'dayofyear_sin', 'dayofyear_cos',
    'pe_0', 'pe_1', 'pe_2', 'pe_3', 
    'pe_4', 'pe_5', 'pe_6', 'pe_7', 
    'pe_8', 'pe_9', 'pe_10', 'pe_11', 
    'pe_12', 'pe_13', 'pe_14', 'pe_15'
]

# Get feature indices for no_scale_features
no_scale_indices = [feature_names.index(f) for f in no_scale_features]

# Scaling PM2.5 and saving its scaler separately
pm25_idx = feature_names.index('PM25')
pm25_scaler = StandardScaler()
pm25_data = imputed_data[:, :, pm25_idx].reshape(-1, 1)
scaled_data[:, :, pm25_idx] = pm25_scaler.fit_transform(pm25_data).reshape(
    imputed_data.shape[0], imputed_data.shape[1]
)
scalers['PM25'] = pm25_scaler

# Scale each feature independently, preserving cyclic features
for feature_idx, feature_name in enumerate(feature_names):
    if feature_name in no_scale_features:
        # Copy cyclic features without scaling
        scaled_data[:, :, feature_idx] = imputed_data[:, :, feature_idx]
    elif feature_name != 'PM25':
        # Scale other features
        scaler = StandardScaler()
        feature_data = imputed_data[:, :, feature_idx].reshape(-1, 1)
        scaled_data[:, :, feature_idx] = scaler.fit_transform(feature_data).reshape(
            imputed_data.shape[0], imputed_data.shape[1]
        )
        scalers[feature_name] = scaler

# Save scalers for later use
# joblib.dump(scalers, 'data/feature_scalers.pkl')

# Print final statistics
print("\nFinal Statistics:")
print(f"Original number of stations: {stations}")
print(f"Number of stations after filtering: {len(filtered_nodes)}")
print(f"Final data shape: {scaled_data.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Number of remaining NaNs: {np.isnan(scaled_data).sum()}")

# Save results
print("Saving processed data...")
# np.save('data/original_pm25.npy', imputed_data[:, :, pm25_idx])
np.save('scaled_data.npy', scaled_data)
np.save('data/lats.npy', lats)
np.save('data/lons.npy', lons)
# np.save('data/nan_position.npy', nan_position)

from vmdpy import VMD
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np

time_steps, stations, features = scaled_data.shape

all_pollutant_features = ['Obs_SO2', 'CO', 'Obs_O3', 'Obs_NO2', 'Obs_PM10', 'PM25']
pm25_feature = ['PM25']
all_pollutant_idx = [feature_names.index(f) for f in all_pollutant_features]
pm25_idx = feature_names.index('PM25')

def apply_vmd(data, feature_idx, K=6, alpha=2000):
    print(f"Applying VMD to {feature_names[feature_idx]} (K={K}, alpha={alpha})...")
    feature_data = data[:, :, feature_idx].reshape(time_steps, stations)
    modes_per_station = []
    
    for station in tqdm(range(stations), desc="Processing stations"):
        signal = feature_data[:, station]
        u, _, _ = VMD(signal, alpha=alpha, tau=0., K=K, DC=0, init=1, tol=1e-7)
        modes_per_station.append(u)  # Shape (K, time_steps)
    return np.array(modes_per_station)  # Shape (stations, K, time_steps)

def apply_pca_station_wise(data, feature_indices, n_components=2):
    print(f"Applying PCA on pollutants (n_components={n_components})...")
    pca_results = np.zeros((time_steps, stations, n_components))
    explained_variances = []
    
    pollutant_data = data[:, :, feature_indices]  # (time_steps, stations, n_pollutants)
    for station in tqdm(range(stations), desc="Processing stations for PCA"):
        station_data = pollutant_data[:, station, :]
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(station_data)
        pca_results[:, station, :] = pca_result
        variance_explained = pca.explained_variance_ratio_
        total_variance = np.sum(variance_explained)
        print(f"Station {station}: Total variance explanined: {total_variance:.4f}")
    return pca_results

# Create combined feature set from original, VMD, and PCA
def create_combined_features(original_data, vmd_modes, pca_features, pollutant_indices):
    keep_indices = [i for i in range(original_data.shape[2]) if i not in pollutant_indices]
    non_pollutant_data = original_data[:, :, keep_indices]
    vmd_modes_reshaped = np.transpose(vmd_modes, (2, 0, 1))
    combined = np.concatenate([non_pollutant_data, vmd_modes_reshaped, pca_features], axis=2)
    return combined

print("\n===== Starting Feature Engineering Process =====")
pm25_vmd_modes = apply_vmd(scaled_data, pm25_idx, K=6, alpha=1000)
print(f"PM2.5 VMD modes shape: {pm25_vmd_modes.shape}")

pca_features = apply_pca_station_wise(scaled_data, all_pollutant_idx, n_components=3)
print(f"PCA features shape: {pca_features.shape}")

scaled_data = scaled_data[:-1]
# pm25_vmd_modes = pm25_vmd_modes[]
pca_features = pca_features[:-1]
print(scaled_data.shape, pm25_vmd_modes.transpose(2, 0, 1).shape, pca_features.shape)

def create_combined_features(original_data, vmd_modes, pca_features, pollutant_indices):
    keep_indices = [i for i in range(original_data.shape[2]) if i not in pollutant_indices]
    non_pollutant_data = original_data[:, :, keep_indices]
    # vmd_modes_reshaped = np.transpose(vmd_modes, (2, 0, 1))
    combined = np.concatenate([non_pollutant_data, vmd_modes, pca_features], axis=2)
    return combined

final_features = create_combined_features(
    scaled_data, 
    pm25_vmd_modes.transpose(2, 0, 1), 
    pca_features, 
    all_pollutant_idx
)

# Create new feature names
non_pollutant_features = [feature_names[i] for i in range(len(feature_names)) 
                          if i not in all_pollutant_idx]
vmd_feature_names = [f'PM25_mode_{i}' for i in range(6)]
pca_feature_names = [f'pollutant_pca_{i}' for i in range(3)]
new_feature_names = non_pollutant_features + vmd_feature_names + pca_feature_names

print(f"\nFinal feature shape: {final_features.shape}")
print(f"Number of new features: {len(new_feature_names)}")
print(f"New feature names: {new_feature_names}")
print("\n===== Feature Engineering Complete =====")
np.save('scaled_data.npy', scaled_data)
np.save('final_features.npy', final_features)



# Final feature shape: (35052, 170, 42)
# Number of new features: 42
# New feature names: ['U_WIND', 'V_WIND', 'WDIR10_cos', 'WDIR10_sin', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'dayofyear_sin', 'dayofyear_cos', 'pe_0', 'pe_1', 'pe_2', 'pe_3', 'pe_4', 'pe_5', 'pe_6', 'pe_7', 'pe_8', 'pe_9', 'pe_10', 'pe_11', 'pe_12', 'pe_13', 'pe_14', 'pe_15', 'PBL', 'TEMP2', 'Q2', 'RN', 'RC', 'PM25_mode_0', 'PM25_mode_1', 'PM25_mode_2', 'PM25_mode_3', 'PM25_mode_4', 'PM25_mode_5', 'pollutant_pca_0', 'pollutant_pca_1', 'pollutant_pca_2']
