import pandas as pd

def verify_data(filename):
    df = pd.read_csv(filename, sep=';', parse_dates=['date_de_tirage'], usecols=[
        'date_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2'
    ], dayfirst=True)
    print("Premières lignes du DataFrame :")
    print(df.head())
    print("\nColonnes du DataFrame :")
    print(df.columns)

if __name__ == "__main__":
    verify_data("euromillions_202002.csv")