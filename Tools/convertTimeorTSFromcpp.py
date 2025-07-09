

from func_standard import timestamp_to_date_utc,date_to_timestamp_utc


import os
import csv
from datetime import datetime
import numpy as np
from numba import njit, prange
import pandas as pd

if __name__ == "__main__":
    # Exemple de conversion de timestamp en date UTC
    timestamp = 1751054395

  # Exemple de timestamp
    formatted_date = timestamp_to_date_utc(timestamp)
    print("Date UTC formatée:", formatted_date)  # Output: "Date UTC formatée: 2021-06-01 00:02:18"
    exit(45)

    # Exemple de conversion de date UTC en timestamp
    year = 2021
    month = 6
    day = 1
    hour = 0
    minute = 2
    second = 18
    timestamp = date_to_timestamp_utc(year, month, day, hour, minute, second)
    print("exemple Timestamp date 20210601:", timestamp)  # Output: "Timestamp: 1622519738"


    from pathlib import Path
    import datetime

    def analyze_candle_files(directory_path):
        """Analyse les fichiers de données de bougies et affiche les timestamps"""

        directory = Path(directory_path)
        files = list(directory.glob("of_raw_candles_dataNew_*.csv"))

        if not files:
            print(f"Aucun fichier correspondant trouvé dans {directory_path}")
            return

        print(f"Nombre de fichiers trouvés: {len(files)}")

        for file_path in sorted(files):
            try:
                # Lecture du fichier CSV
                df = pd.read_csv(file_path, sep=';')

                if 'timeStampOpening' not in df.columns:
                    print(f"Fichier {file_path.name}: La colonne 'timeStampOpening' est manquante")
                    continue

                # Récupération des timestamps
                first_timestamp = df['timeStampOpening'].iloc[0]
                last_timestamp = df['timeStampOpening'].iloc[-1]

                # Conversion en date
                first_date = timestamp_to_date_utc(first_timestamp)
                last_date = timestamp_to_date_utc(last_timestamp)

                # Affichage des résultats
                print(f"\nFichier: {file_path.name}")
                print(f"Nombre de lignes: {len(df)}")
                print(f"Premier timestamp: {first_timestamp} → Date UTC: {first_date}")
                print(f"Dernier timestamp: {last_timestamp} → Date UTC: {last_date}")

                # Calculer la durée couverte
                time_span = last_timestamp - first_timestamp
                if len(str(int(first_timestamp))) > 10:  # Si en millisecondes
                    days = time_span / (1000 * 60 * 60 * 24)
                else:
                    days = time_span / (60 * 60 * 24)

                print(f"Période couverte: {days:.2f} jours")

            except Exception as e:
                print(f"Erreur lors de l'analyse du fichier {file_path.name}: {e}")


    # Chemin vers le répertoire contenant les fichiers
    directory_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2"

    # Exécution de l'analyse
    analyze_candle_files(directory_path)

    import pandas as pd
    import os
    import re
    from datetime import datetime


    def verify_timestamp_sequence(directory, file_pattern="Step1_*.csv"):
        """
        Vérifie que:
        1. Les timestamps à l'intérieur de chaque fichier sont croissants ou égaux
        2. L'enchaînement des timestamps entre fichiers consécutifs (par ordre de suffixe)
           préserve également cette propriété
        """
        # Récupérer tous les fichiers correspondant au modèle
        all_files = [f for f in os.listdir(directory) if f.startswith("Step1_") and f.endswith(".csv")]

        # Extraire le numéro de suffixe pour chaque fichier
        file_numbers = {}
        for file in all_files:
            match = re.search(r'_(\d+)\.csv$', file)
            if match:
                suffix = int(match.group(1))
                file_numbers[file] = suffix

        # Trier les fichiers par leur suffixe
        sorted_files = sorted(all_files, key=lambda f: file_numbers.get(f, float('inf')))

        print(f"Fichiers trouvés et triés par suffixe:")
        for i, file in enumerate(sorted_files):
            print(f"  {i + 1}. {file} (suffixe: _{file_numbers[file]})")

        # Analyser chaque fichier dans l'ordre des suffixes
        file_info = []

        for file in sorted_files:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, sep=';')

            if 'timeStampOpening' not in df.columns:
                print(f"\nERREUR: La colonne 'timeStampOpening' n'existe pas dans {file}")
                continue

            # Convertir en numérique si nécessaire
            df['timeStampOpening'] = pd.to_numeric(df['timeStampOpening'])

            # Vérifier si les timestamps sont croissants ou égaux (non décroissants)
            is_monotonic = df['timeStampOpening'].is_monotonic_increasing

            # Vérifier s'il y a des timestamps strictement décroissants
            decreasing_instances = 0
            prev_ts = df['timeStampOpening'].iloc[0]
            decreasing_examples = []

            for i in range(1, len(df)):
                curr_ts = df['timeStampOpening'].iloc[i]
                if curr_ts < prev_ts:
                    decreasing_instances += 1
                    decreasing_examples.append((i, prev_ts, curr_ts))
                    if len(decreasing_examples) >= 3:  # Limiter à 3 exemples
                        break
                prev_ts = curr_ts

            # Obtenir les premiers et derniers timestamps
            first_timestamp = df['timeStampOpening'].iloc[0]
            last_timestamp = df['timeStampOpening'].iloc[-1]

            # Convertir timestamps en date lisible
            first_date = datetime.utcfromtimestamp(first_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            last_date = datetime.utcfromtimestamp(last_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            file_info.append({
                'file': file,
                'suffix': file_numbers[file],
                'rows': len(df),
                'first_timestamp': first_timestamp,
                'last_timestamp': last_timestamp,
                'first_date': first_date,
                'last_date': last_date,
                'is_monotonic': is_monotonic,
                'decreasing_instances': decreasing_instances
            })

            print(f"\nFichier {file}:")
            print(f"  Suffixe: _{file_numbers[file]}")
            print(f"  Timestamps en ordre croissant ou égal: {is_monotonic}")
            print(f"  Nombre d'instances où timestamp[i] < timestamp[i-1]: {decreasing_instances}")
            print(f"  Nombre de lignes: {len(df)}")
            print(f"  Premier timestamp: {first_timestamp} ({first_date})")
            print(f"  Dernier timestamp: {last_timestamp} ({last_date})")

            if decreasing_instances > 0:
                print("  PROBLÈME: Timestamps décroissants détectés dans ce fichier!")
                print("  Exemples de timestamps décroissants:")
                for idx, prev, curr in decreasing_examples:
                    prev_date = datetime.utcfromtimestamp(prev).strftime('%Y-%m-%d %H:%M:%S')
                    curr_date = datetime.utcfromtimestamp(curr).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"    Ligne {idx + 1}: {curr} ({curr_date}) < ligne {idx}: {prev} ({prev_date})")

        # Vérifier l'enchaînement des timestamps entre fichiers consécutifs
        print("\n=== VÉRIFICATION DE L'ENCHAÎNEMENT DES TIMESTAMPS ENTRE FICHIERS ===")
        sequence_ok = True

        # Vérifier si chaque fichier a des timestamps croissants ou égaux en interne
        all_files_monotonic = all(info['is_monotonic'] for info in file_info)

        if not all_files_monotonic:
            print("ATTENTION: Certains fichiers ont des timestamps non croissants en interne.")
            print("           L'enchaînement global ne peut pas être garanti.")
            sequence_ok = False

        # Vérifier les transitions entre fichiers consécutifs
        for i in range(len(file_info) - 1):
            current_file = file_info[i]
            next_file = file_info[i + 1]

            print(f"\nTransition {current_file['file']} → {next_file['file']} : ", end="")

            if next_file['first_timestamp'] < current_file['last_timestamp']:
                print("PROBLÈME")
                print(
                    f"  Dernier timestamp de {current_file['file']}: {current_file['last_timestamp']} ({current_file['last_date']})")
                print(
                    f"  Premier timestamp de {next_file['file']}: {next_file['first_timestamp']} ({next_file['first_date']})")
                print(
                    f"  Le premier timestamp du fichier suivant est ANTÉRIEUR au dernier timestamp du fichier précédent!")
                sequence_ok = False
            else:
                print("OK")
                print(
                    f"  Dernier timestamp de {current_file['file']}: {current_file['last_timestamp']} ({current_file['last_date']})")
                print(
                    f"  Premier timestamp de {next_file['file']}: {next_file['first_timestamp']} ({next_file['first_date']})")

        # Conclusion
        print("\n=== CONCLUSION ===")
        if all_files_monotonic:
            print("✅ Tous les fichiers ont leurs timestamps en ordre croissant ou égal en interne.")
        else:
            print("❌ Certains fichiers ont des timestamps non croissants en interne.")

        if sequence_ok and all_files_monotonic:
            print("✅ L'enchaînement des timestamps entre fichiers consécutifs est correct.")
            print("✅ RÉSULTAT FINAL: La séquence complète des timestamps est croissante ou égale.")
        else:
            print("❌ L'enchaînement des timestamps entre fichiers consécutifs présente des problèmes.")
            print("❌ RÉSULTAT FINAL: La séquence complète des timestamps N'EST PAS croissante ou égale.")

            if not all_files_monotonic:
                print(
                    "\nSolution recommandée: Corriger l'ordre des timestamps à l'intérieur des fichiers problématiques.")
            else:
                print("\nSolutions possibles:")
                print("1. Renommer les fichiers pour que leur suffixe corresponde à l'ordre chronologique")
                print("2. Modifier le code de fusion pour trier les fichiers par leur premier timestamp")
                print("   au lieu de les trier par suffixe")

        return sequence_ok and all_files_monotonic


    # Chemin du répertoire
    directory = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\\merge"

    # Exécuter la vérification
    sequence_valid = verify_timestamp_sequence(directory)