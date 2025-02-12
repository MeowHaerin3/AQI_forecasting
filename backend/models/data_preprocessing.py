import pandas as pd
import numpy as np
import aqi


def remove_whitespace_cols(df):
    df.columns = df.columns.str.strip()
    return df

def cleaning_data(df):
    df = remove_whitespace_cols(df)
    if 'date' not in df.columns:
        raise ValueError('date column not found')
    else:
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    return df

def calc_aqi(row: pd.Series):
    pollutants = {
        'pm25': aqi.POLLUTANT_PM25,
        'pm10': aqi.POLLUTANT_PM10,
        'o3': aqi.POLLUTANT_O3_8H,
        'no2': aqi.POLLUTANT_NO2_1H,
        'so2': aqi.POLLUTANT_SO2_1H,
        'co': aqi.POLLUTANT_CO_8H
    }
    aqi_values = []
    for pollutant, aqi_pollutant in pollutants.items():
        if pollutant in row and pd.notna(row[pollutant]):
            try:
                aqi_values.append(aqi.to_aqi([(aqi_pollutant, row[pollutant])]))
            except TypeError:
                continue
    if len(aqi_values) == 0:
        return np.nan
    return np.max(aqi_values)

def preprocess_data(df):
    
    if 'date' not in df.columns or 'aqi' not in df.columns:
        raise ValueError('Required columns not found')
    aqi_df = df[['date', 'aqi']].copy()
    aqi_df['aqi'] = pd.to_numeric(aqi_df['aqi'], errors='coerce')
    aqi_df.set_index('date', inplace=True)
    
    aqi_df['aqi'] = aqi_df['aqi'].interpolate(method='time')
    return aqi_df

def load_and_clean_data(data_path):
    df = pd.read_csv(data_path)
    df_clean = cleaning_data(df)
    df_clean['aqi'] = df_clean.apply(calc_aqi, axis=1)
    aqi_df = preprocess_data(df_clean)
    return aqi_df

