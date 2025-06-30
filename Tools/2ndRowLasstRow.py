import csv
import os
import re
from standardFunc_sauv import timestamp_to_date_utc, date_to_timestamp_utc
import pandas as pd
from datetime import datetime
def detect_column_type(column):
    try:
        pd.to_numeric(column)
        return 'float64'
    except ValueError:
        return 'object'
file_name = "of_raw_candles_dataNew_3.csv"
directory_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_5TP_0SL_17102024"

# Construire le chemin complet du fichier d'entrée
file_path = os.path.join(directory_path, file_name)

# Lire le fichier CSV avec détection automatique des types de données
print("Début du chargement des données")
df = pd.read_csv(file_path, sep=';', encoding='iso-8859-1', dtype='object')
print("Fin du chargement des données")
"""
# Supprimer les 900 dernières lignes
df = df.iloc[:-900]

# Construire le chemin complet du fichier de sortie
output_file_name = "of_raw_candles_dataNew_3.csv"
output_file_path = os.path.join(directory_path, output_file_name)

# Sauvegarder le DataFrame dans un nouveau fichier CSV
df.to_csv(output_file_path, sep=';', encoding='iso-8859-1', index=False)
print(f"DataFrame sauvegardé dans {output_file_path}")
"""

# Convertir les colonnes en types appropriés
for col in df.columns:
    df[col] = df[col].astype(detect_column_type(df[col]))

# Obtenir les timestamps
second_row_timestamp = df.iloc[1, 0]  # Deuxième ligne, première colonne
last_row_timestamp = df.iloc[-1, 0]  # Dernière ligne, première colonne

# Convertir les timestamps en format datetime lisible
formatted_datSecond = timestamp_to_date_utc(second_row_timestamp)
formatted_dateLast = timestamp_to_date_utc(last_row_timestamp)


print(f"\nTimestamp de la deuxième ligne : {second_row_timestamp} ({formatted_datSecond})")
print(f"Timestamp de la dernière ligne : {last_row_timestamp} ({formatted_dateLast})")

max_value = df['deltaTimestampOpeningSection5index'].max()
print(f"La valeur maximale de 'deltaTimestampOpeningSection5index' est : {max_value}")
max_value = df['deltaTimestampOpening'].max()
print(f"La valeur maximale de 'deltaTimestampOpening' est : {max_value}")



# Afficher des informations sur les types de données des colonnes
print("\nTypes de données des colonnes:")
print(df.dtypes)

# Afficher des statistiques de base pour les colonnes numériques
print("\nStatistiques de base pour les colonnes numériques:")
print(df.describe())

# Vérifier les valeurs uniques dans la colonne SessionStartEnd
print("\nValeurs uniques dans SessionStartEnd:")
print(df['SessionStartEnd'].unique())

# Vérifier les valeurs de SessionStartEnd au début et à la fin du fichier
first_session_start = df.iloc[0]['SessionStartEnd']
last_session_start = df.iloc[-1]['SessionStartEnd']
print(f"\nPremière valeur de SessionStartEnd: {first_session_start}")
print(f"Dernière valeur de SessionStartEnd: {last_session_start}")



print(f"Le fichier commence par {first_session_start} dans SessionStartEnd")
print(f"Le fichier se termine par {last_session_start} dans SessionStartEnd")

if first_session_start == 10 and last_session_start == 20:
    print("Le fichier commence correctement par 10 et se termine par 20 dans SessionStartEnd.")
else:
    if first_session_start != 10:
        print(f"Attention: Le fichier ne commence pas par 10 dans SessionStartEnd, mais par {first_session_start}")
    if last_session_start != 20:
        print(f"Attention: Le fichier ne se termine pas par 20 dans SessionStartEnd, mais par {last_session_start}")