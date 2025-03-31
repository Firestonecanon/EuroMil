import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def predict_next_draw(model, scaler, targets):
    next_date = pd.Timestamp.now().normalize()
    next_features = pd.DataFrame([[next_date.year, next_date.month, next_date.day,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                 columns=['year', 'month', 'day',
                                          'retard_boule_1', 'retard_boule_2', 'retard_boule_3',
                                          'retard_boule_4', 'retard_boule_5',
                                          'retard_etoile_1', 'retard_etoile_2',
                                          'freq_boule_1', 'freq_boule_2', 'freq_boule_3',
                                          'freq_boule_4', 'freq_boule_5',
                                          'freq_etoile_1', 'freq_etoile_2',
                                          'pattern_pairs'])
    next_features = scaler.transform(next_features)
    next_features = np.reshape(next_features, (next_features.shape[0], 1, next_features.shape[1]))
    
    prediction = model.predict(next_features)
    predicted_numbers = {column: int(np.round(pred)) for column, pred in zip(targets.columns, prediction[0])}
    
    # Assurer que les numéros ne se répètent pas
    predicted_set = set()
    for key in ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']:
        while predicted_numbers[key] in predicted_set or predicted_numbers[key] < 1 or predicted_numbers[key] > 50:
            predicted_numbers[key] = (predicted_numbers[key] + 1) % 50  # Assuming the numbers range from 1 to 50
        predicted_set.add(predicted_numbers[key])
    
    star_set = set()
    for key in ['etoile_1', 'etoile_2']:
        while predicted_numbers[key] in star_set or predicted_numbers[key] < 1 or predicted_numbers[key] > 12:
            predicted_numbers[key] = (predicted_numbers[key] + 1) % 12  # Assuming the star numbers range from 1 to 12
        star_set.add(predicted_numbers[key])
    
    print(f"Prediction for next draw: {predicted_numbers}")

if __name__ == "__main__":
    from train_model import train_model
    from prepare_data import prepare_data
    features, targets = prepare_data("euromillions_202002.csv")
    scaler = MinMaxScaler().fit(features)
    
    # Fit scaler with feature names
    scaler.fit(pd.DataFrame(features, columns=['year', 'month', 'day',
                                               'retard_boule_1', 'retard_boule_2', 'retard_boule_3',
                                               'retard_boule_4', 'retard_boule_5',
                                               'retard_etoile_1', 'retard_etoile_2',
                                               'freq_boule_1', 'freq_boule_2', 'freq_boule_3',
                                               'freq_boule_4', 'freq_boule_5',
                                               'freq_etoile_1', 'freq_etoile_2',
                                               'pattern_pairs']))
    model = train_model(features, targets)
    predict_next_draw(model, scaler, targets)