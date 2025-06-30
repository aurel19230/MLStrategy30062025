import numpy as np
import pandas as pd
import os
import time
pd.set_option("display.max_columns", None)
from standardFunc_sauv import timestamp_to_date_utc,calculate_naked_poc_distances

if __name__ == '__main__':
    # Configuration
    FILE_NAME = "Step3_4_0_6TP_1SL_080919_141024_extractOnlyFullSession.csv"
    DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\merge"
    FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME)

    # Chargement des données
    print("Loading data...")
    start_time = time.time()
    df = pd.read_csv(FILE_PATH, sep=";", encoding="ISO-8859-1")
    load_time = time.time() - start_time
    print(f"Loaded time: {load_time:.2f} seconds")
    print("Data loaded. Shape:", df.shape)

    # Calcul des distances avec timing
    print("\nCalculating naked POC distances...")
    start_time = time.time()
    dist_above, dist_below = calculate_naked_poc_distances(df)
    calc_time = time.time() - start_time
    print(f"Calculation time: {calc_time:.2f} seconds")

    # Ajout des colonnes au DataFrame
    print("\nUpdating DataFrame...")
    df['naked_poc_dist_above'] = dist_above
    df['naked_poc_dist_below'] = dist_below
    df['date_utc'] = df['timeStampOpening'].apply(timestamp_to_date_utc)

    # Affichage des premiers résultats
    print("\nFirst 5 rows:")
    for idx, row in df.head().iterrows():
        print(
            f"Index {idx} - "
            f"Date UTC: {row['date_utc']} - "
            f"Above POC: {row['naked_poc_dist_above']:.2f} - "
            f"Below POC: {row['naked_poc_dist_below']:.2f}"
        )

    print(f"\nTotal processing time: {load_time + calc_time:.2f} seconds")