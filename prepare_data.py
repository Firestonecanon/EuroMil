import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(filename):
    df = pd.read_csv(filename, sep=';', parse_dates=['date_de_tirage'], usecols=[
        'date_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2'
    ], dayfirst=True)
    df.sort_values(by='date_de_tirage', inplace=True)
    df['year'] = df['date_de_tirage'].dt.year
    df['month'] = df['date_de_tirage'].dt.month
    df['day'] = df['date_de_tirage'].dt.day
    
    for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']:
        df[f'retard_{col}'] = df.apply(lambda row: calculate_delay(row, col, df), axis=1)
        df[f'freq_{col}'] = df.apply(lambda row: calculate_frequency(row, col, df), axis=1)
    
    df['pattern_pairs'] = df.apply(lambda row: calculate_pattern(row), axis=1)
    
    features = df[['year', 'month', 'day'] +
                  [f'retard_{col}' for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']] +
                  [f'freq_{col}' for col in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']] +
                  ['pattern_pairs']]
    targets = df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']]
    
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    return features, targets

def calculate_delay(row, column, df):
    current_index = row.name
    current_value = row[column]
    previous_draws = df.loc[:current_index-1]

    if previous_draws.empty:
        return 0
    
    last_occurrence = previous_draws[previous_draws[column] == current_value]
    
    if last_occurrence.empty:
        return len(previous_draws)
    
    return current_index - last_occurrence.index[-1]

def calculate_frequency(row, column, df):
    current_index = row.name
    current_value = row[column]
    previous_draws = df.loc[:current_index-1]
    
    if previous_draws.empty:
        return 0
    
    frequency = previous_draws[column].value_counts().get(current_value, 0)
    return frequency

def calculate_pattern(row):
    nums = [row['boule_1'], row['boule_2'], row['boule_3'], row['boule_4'], row['boule_5']]
    pairs = sum(1 for num in nums if num % 2 == 0)
    return pairs

if __name__ == "__main__":
    features, targets = prepare_data("euromillions_202002.csv")
    print("Features shape:", features.shape)
    print("Targets shape:", targets.shape)