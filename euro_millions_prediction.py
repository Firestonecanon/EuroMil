import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.constraints import MaxNorm

class EuroMillionsPredictor:
    def __init__(self, filename):
        # Lire le fichier CSV en spécifiant les colonnes nécessaires
        self.df = pd.read_csv(filename, sep=';', parse_dates=['date_de_tirage'], usecols=[
            'date_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2'
        ], dayfirst=True)
        self.check_data()  # Vérifier les données lues
        self.prepare_data()

    def check_data(self):
        """Vérifie les premières lignes du DataFrame pour s'assurer que le fichier est bien lu"""
        print("Premières lignes du DataFrame :")
        print(self.df.head())
        print("\nColonnes du DataFrame :")
        print(self.df.columns)

    def prepare_data(self):
        """Prépare les données pour l'entraînement du modèle"""
        self.df.sort_values(by='date_de_tirage', inplace=True)
        self.df['year'] = self.df['date_de_tirage'].dt.year
        self.df['month'] = self.df['date_de_tirage'].dt.month
        self.df['day'] = self.df['date_de_tirage'].dt.day
        
        # Ajouter les retards, fréquences et motifs comme caractéristiques
        self.df['retard_boule_1'] = self.df.apply(lambda row: self.calculate_delay(row, 'boule_1'), axis=1)
        self.df['retard_boule_2'] = self.df.apply(lambda row: self.calculate_delay(row, 'boule_2'), axis=1)
        self.df['retard_boule_3'] = self.df.apply(lambda row: self.calculate_delay(row, 'boule_3'), axis=1)
        self.df['retard_boule_4'] = self.df.apply(lambda row: self.calculate_delay(row, 'boule_4'), axis=1)
        self.df['retard_boule_5'] = self.df.apply(lambda row: self.calculate_delay(row, 'boule_5'), axis=1)
        self.df['retard_etoile_1'] = self.df.apply(lambda row: self.calculate_delay(row, 'etoile_1'), axis=1)
        self.df['retard_etoile_2'] = self.df.apply(lambda row: self.calculate_delay(row, 'etoile_2'), axis=1)
        
        self.df['freq_boule_1'] = self.df.apply(lambda row: self.calculate_frequency(row, 'boule_1'), axis=1)
        self.df['freq_boule_2'] = self.df.apply(lambda row: self.calculate_frequency(row, 'boule_2'), axis=1)
        self.df['freq_boule_3'] = self.df.apply(lambda row: self.calculate_frequency(row, 'boule_3'), axis=1)
        self.df['freq_boule_4'] = self.df.apply(lambda row: self.calculate_frequency(row, 'boule_4'), axis=1)
        self.df['freq_boule_5'] = self.df.apply(lambda row: self.calculate_frequency(row, 'boule_5'), axis=1)
        self.df['freq_etoile_1'] = self.df.apply(lambda row: self.calculate_frequency(row, 'etoile_1'), axis=1)
        self.df['freq_etoile_2'] = self.df.apply(lambda row: self.calculate_frequency(row, 'etoile_2'), axis=1)
        
        self.df['pattern_pairs'] = self.df.apply(lambda row: self.calculate_pattern(row), axis=1)
        
        self.features = self.df[['year', 'month', 'day',
                                 'retard_boule_1', 'retard_boule_2', 'retard_boule_3',
                                 'retard_boule_4', 'retard_boule_5',
                                 'retard_etoile_1', 'retard_etoile_2',
                                 'freq_boule_1', 'freq_boule_2', 'freq_boule_3',
                                 'freq_boule_4', 'freq_boule_5',
                                 'freq_etoile_1', 'freq_etoile_2',
                                 'pattern_pairs']]
        self.targets = self.df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']]
        
        # Normaliser les caractéristiques
        self.scaler = MinMaxScaler()
        self.features = self.scaler.fit_transform(self.features)
    
    def calculate_delay(self, row, column):
        """Calcule le retard d'un numéro ou d'une étoile"""
        current_index = row.name
        current_value = row[column]
        previous_draws = self.df.loc[:current_index-1]
        
        if previous_draws.empty:
            return 0
        
        last_occurrence = previous_draws[previous_draws[column] == current_value]
        
        if last_occurrence.empty:
            return len(previous_draws)
        
        return current_index - last_occurrence.index[-1]
    
    def calculate_frequency(self, row, column):
        """Calcule la fréquence d'apparition d'un numéro ou d'une étoile"""
        current_index = row.name
        current_value = row[column]
        previous_draws = self.df.loc[:current_index-1]
        
        if previous_draws.empty:
            return 0
        
        frequency = previous_draws[column].value_counts().get(current_value, 0)
        return frequency
    
    def calculate_pattern(self, row):
        """Calcule le motif pairs/impairs d'un tirage"""
        nums = [row['boule_1'], row['boule_2'], row['boule_3'], row['boule_4'], row['boule_5']]
        pairs = sum(1 for num in nums if num % 2 == 0)
        return pairs
    
    def train_model(self):
        """Entraîne un modèle LSTM"""
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.targets, test_size=0.2, random_state=42)
        
        # Reshape les données pour LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, input_shape=(1, X_train.shape[2]), kernel_constraint=MaxNorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, kernel_constraint=MaxNorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(7))
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        
        y_pred = self.model.predict(X_test)
        for i, column in enumerate(self.targets.columns):
            accuracy = np.mean(np.round(y_pred[:, i]) == y_test[column])
            print(f"Accuracy for {column}: {accuracy * 100:.2f}%")
        
    def predict_next_draw(self):
        """Prévoit le prochain tirage des EuroMillions"""
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
        next_features = self.scaler.transform(next_features)
        next_features = np.reshape(next_features, (next_features.shape[0], 1, next_features.shape[1]))
        
        prediction = self.model.predict(next_features)
        predicted_numbers = {column: int(np.round(pred)) for column, pred in zip(self.targets.columns, prediction[0])}
        
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
    predictor = EuroMillionsPredictor("euromillions_202002.csv")
    predictor.train_model()
    predictor.predict_next_draw()