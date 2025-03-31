import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EuroMillionsAnalyzer:
    def __init__(self, filename):
        # Lire le fichier CSV avec le séparateur correct
        self.df = pd.read_csv(filename, sep=';', parse_dates=['date_de_tirage'])
        
        # Trier les données par date_de_tirage
        self.df.sort_values(by='date_de_tirage', inplace=True)
        
        # Afficher les colonnes du DataFrame pour vérifier leur présence et leur orthographe
        print("Colonnes présentes dans le fichier CSV :")
        print(self.df.columns)
        
        # Afficher les premières lignes du DataFrame pour vérification
        print("\nPremières lignes du DataFrame :")
        print(self.df.head())
        
        self.prepare_data()
        
    def prepare_data(self):
        """Prépare toutes les données pour l'analyse"""
        # Numéros et étoiles
        self.all_numbers = self.df[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values
        self.all_stars = self.df[['etoile_1', 'etoile_2']].values
        
        # Pour l'analyse des retards
        self.last_seen_numbers = {num: float('inf') for num in range(1, 51)}
        self.last_seen_stars = {star: float('inf') for star in range(1, 13)}
        self.update_last_seen()
        
    def update_last_seen(self):
        """Calcule le nombre de tirages depuis la dernière apparition"""
        for i, row in self.df.iterrows():
            for num in range(1, 51):
                self.last_seen_numbers[num] += 1
            for star in range(1, 13):
                self.last_seen_stars[star] += 1
                
            for num in row[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']]:
                self.last_seen_numbers[num] = 0
            for star in row[['etoile_1', 'etoile_2']]:
                self.last_seen_stars[star] = 0
    
    def analyze_frequencies(self):
        """Analyse les fréquences simples"""
        numbers = self.all_numbers.flatten()
        stars = self.all_stars.flatten()
        
        print("\nTOP 15 Numéros:")
        for num, count in Counter(numbers).most_common(15):
            print(f"{num:2d}: {count:3d} fois | {count/len(self.df)*100:.1f}%")
            
        print("\nTOP 10 Étoiles:")
        for star, count in Counter(stars).most_common(10):
            print(f"{star:2d}: {count:3d} fois | {count/len(self.df)*100:.1f}%")
    
    def analyze_retards(self):
        """Identifie les numéros et étoiles en retard"""
        print("\nNuméros les plus en retard:")
        for num, retard in sorted(self.last_seen_numbers.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Numéro {num:2d}: {retard} tirages sans apparition")
            
        print("\nÉtoiles les plus en retard:")
        for star, retard in sorted(self.last_seen_stars.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"Étoile {star:2d}: {retard} tirages sans apparition")
    
    def analyze_combinations(self, n=2):
        """Analyse les combinaisons de n numéros"""
        comb_counts = defaultdict(int)
        
        for draw in self.all_numbers:
            for comb in combinations(sorted(draw), n):
                comb_counts[comb] += 1
                
        print(f"\nTOP 20 Combinaisons de {n} numéros:")
        for comb, count in sorted(comb_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"{'-'.join(map(str, comb))}: {count} fois")
    
    def analyze_patterns(self):
        """Analyse les motifs (pairs/impairs, etc.)"""
        patterns = []
        for draw in self.all_numbers:
            pairs = sum(1 for num in draw if num % 2 == 0)
            patterns.append((pairs, 5-pairs))
            
        pattern_counts = Counter(patterns)
        print("\nMotifs Pairs/Impairs:")
        for (pairs, impairs), count in pattern_counts.most_common():
            print(f"{pairs}P-{impairs}I: {count:3d} fois | {count/len(self.df)*100:.1f}%")
    
    def generate_recommendation(self):
        """Génère une recommandation de numéros"""
        # Stratégie hybride
        hot_numbers = [num for num, _ in Counter(self.all_numbers.flatten()).most_common(15)]
        cold_numbers = [num for num, _ in sorted(self.last_seen_numbers.items(), key=lambda x: x[1], reverse=True)[:15]]
        
        recommended = list(set(hot_numbers[:8] + cold_numbers[:7]))
        np.random.shuffle(recommended)
        
        print("\n💡 Recommandation stratégique:")
        print(f"Numéros: {' '.join(map(str, sorted(recommended[:5])))})")
        print(f"Étoiles: {' '.join(map(str, sorted(np.random.choice(list(self.last_seen_stars.keys()), 2, replace=False))))}")
    
    def full_analysis(self):
        """Exécute toutes les analyses"""
        print(f"\n{' ANALYSE COMPLÈTE ':=^80}")
        print(f"Période: {self.df['date_de_tirage'].min().strftime('%d/%m/%Y')} -> {self.df['date_de_tirage'].max().strftime('%d/%m/%Y')}")
        print(f"Nombre de tirages analysés: {len(self.df)}\n")
        
        self.analyze_frequencies()
        self.analyze_retards()
        self.analyze_combinations(2)
        self.analyze_combinations(3)
        self.analyze_patterns()
        self.generate_recommendation()
        
        # Visualisations
        self.plot_frequencies()
        
    def plot_frequencies(self):
        """Affiche les visualisations"""
        plt.figure(figsize=(18, 12))
        
        # Fréquence des numéros
        plt.subplot(2, 2, 1)
        nums = pd.Series(self.all_numbers.flatten())
        nums.value_counts().sort_index().plot.bar(color='blue')
        plt.title("Fréquence des numéros principaux")
        
        # Fréquence des étoiles
        plt.subplot(2, 2, 2)
        stars = pd.Series(self.all_stars.flatten())
        stars.value_counts().sort_index().plot.bar(color='gold')
        plt.title("Fréquence des étoiles")
        
        # Retard des numéros
        plt.subplot(2, 2, 3)
        pd.Series(self.last_seen_numbers).sort_index().plot.bar(color='red')
        plt.title("Nombre de tirages depuis dernière apparition (Numéros)")
        
        # Retard des étoiles
        plt.subplot(2, 2, 4)
        pd.Series(self.last_seen_stars).sort_index().plot.bar(color='orange')
        plt.title("Nombre de tirages depuis dernière apparition (Étoiles)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("=== SUPER ANALYSEUR EUROMILLIONS ===")
    analyzer = EuroMillionsAnalyzer("euromillions_202002.csv")
    analyzer.full_analysis()
    input("\nAppuyez sur Entrée pour quitter...")