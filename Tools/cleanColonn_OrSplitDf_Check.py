from standardFunc_sauv import load_data, split_sessions, print_notification,calculate_and_display_sessions
import os
import pandas as pd
import numpy as np
from numba import njit, prange
import time

FILE_NAME_ = "Step1_150322_091022_4TicksRev_1.csv"
DIRECTORY_PATH_ = file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_04102024\merge"
FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)
option = input("Appuyez sur 'c' pour ajouter la fin de session, \nou 's' pour diviser le fichier, \n ou 'a' analiser les suite de sessionStartEnd, "
               "\nou 't' pour calculer le Nombre de session: ").lower()

# Chargement des données
variables = [
    "timeStampOpening", "deltaTimestampOpening", "deltaTimestampOpeningSection", "SessionStartEnd",
    "indexTrade_result", "candleDir", "candleSizeTicks", "close", "open", "high", "low",
    "pocPrice", "volPOC", "deltaPOC", "volume", "delta",
    "VolBlw", "DeltaBlw", "VolAbv", "DeltaAbv",
    "VolBlw_6Tick", "DeltaBlw_6Tick", "VolAbv_6Tick", "DeltaAbv_6Tick",
    "bidVolLow", "askVolLow", "bidVolLow_1", "askVolLow_1", "bidVolLow_2", "askVolLow_2",
    "bidVolLow_3", "askVolLow_3",
    "bidVolHigh", "askVolHigh", "bidVolHigh_1", "askVolHigh_1", "bidVolHigh_2", "askVolHigh_2",
    "bidVolHigh_3", "askVolHigh_3",
    "VWAP", "VWAPsd1Top", "VWAPsd2Top", "VWAPsd3Top", "VWAPsd4Top",
    "VWAPsd1Bot", "VWAPsd2Bot", "VWAPsd3Bot", "VWAPsd4Bot",
    "bandWidthBB", "perctBB", "atr",
    "vaH_6periods", "vaPoc_6periods", "vaL_6periods", "vaVol_6periods", "vaDelta_6periods",
    "vaH_11periods", "vaPoc_11periods", "vaL_11periods", "vaVol_11periods", "vaDelta_11periods",
    "vaH_16periods", "vaPoc_16periods", "vaL_16periods", "vaVol_16periods", "vaDelta_16periods"
]




@njit
def check_session_start_end(session_start_end, time_stamp_opening):
    for i in range(len(session_start_end) - 1):  # Exclure la dernière ligne
        current_value = session_start_end[i]
        next_value = session_start_end[i + 1]

        if current_value == 20 and next_value != 10:
            timeStampOpening = time_stamp_opening[i]
            print(f"Condition non respectée à l'index {i}: SessionStartEnd = 20, mais la prochaine n'est pas 10.")
            print(f"timeStampOpening correspondant : {timeStampOpening}")

nombre_variables = len(variables)

# Chargement du DataFrame
df = load_data(FILE_PATH_)
# Conversion des colonnes pandas en tableaux NumPy pour une compatibilité avec Numba
session_start_end = df['SessionStartEnd'].astype(np.int32).values  # Convertir en entier pour Numba
time_stamp_opening = df['timeStampOpening'].astype(np.int64).values  # Assurer que ce soit en int64 pour Numba
# Nombre de colonnes que vous voulez renommer
n = len(variables)

# Remplacer les noms des n premières colonnes du DataFrame
df.columns = variables[:n] + list(df.columns[n:])


# Fonction Numba pour appliquer la logique 20-10
def apply_20_10_logic(data):
    result = data.copy()
    for i in prange(3, len(data)):
        if int(data[i, 3]) == 10:  # Assurez-vous que 3 est l'index correct pour 'deltaTimestampOpeningSection'
            result[i - 1, 3] = 20
    return result


# Convertir le DataFrame en un array NumPy
data_array = df.values

# Appliquer la logique 20-10 avec Numba
result_array = apply_20_10_logic(data_array)

# Convertir le résultat en DataFrame
df_result = pd.DataFrame(result_array, columns=df.columns)


# Fonction pour ajouter la fin de session


# Fonction pour diviser le fichier en deux
def split_file(df):
    total_rows = len(df)
    middle_index = total_rows // 2
    split_index = df.iloc[middle_index:]['SessionStartEnd'].eq(20).idxmax()
    df1 = df.iloc[:split_index + 1]
    df2 = df.iloc[split_index + 1:]
    return df1, df2


if option == 'c':
    output_file_name = "Cleaned_" + FILE_NAME_
    output_file_path = os.path.join(DIRECTORY_PATH_, output_file_name)
    df_result.to_csv(output_file_path, sep=';', index=False, encoding='iso-8859-1')
    print(f"Le fichier a été écrit avec succès : {output_file_path}")

    # Écrire les noms de colonnes dans un fichier séparé
    column_names_file = "Column_Names_" + FILE_NAME_
    column_names_path = os.path.join(DIRECTORY_PATH_, column_names_file)
    pd.DataFrame({'column_names': df_result.columns}).to_csv(column_names_path, sep=';', index=False,
                                                             encoding='iso-8859-1')
    print(f"Les noms de colonnes ont été écrits dans : {column_names_path}")

elif option == 's':
    df1, df2 = split_file(df_result)

    if df2.iloc[0]['SessionStartEnd'] != 10:
        print(
            "Erreur : Le deuxième fichier ne commence pas par un 10 dans SessionStartEnd. Impossible de diviser le fichier correctement.")
    else:
        output_file_name1 = os.path.splitext(FILE_NAME_)[0] + "_bis1.csv"
        output_file_name2 = os.path.splitext(FILE_NAME_)[0] + "_bis2.csv"
        output_file_path1 = os.path.join(DIRECTORY_PATH_, output_file_name1)
        output_file_path2 = os.path.join(DIRECTORY_PATH_, output_file_name2)

        df1.to_csv(output_file_path1, sep=';', index=False, encoding='iso-8859-1')
        df2.to_csv(output_file_path2, sep=';', index=False, encoding='iso-8859-1')

        print(f"Les fichiers ont été écrits avec succès :")
        print(f"1. {output_file_path1}")
        print(f"2. {output_file_path2}")

        if df1.iloc[0]['SessionStartEnd'] != 10 or df1.iloc[-1]['SessionStartEnd'] != 20:
            print(
                f"Attention : Le premier fichier ({output_file_path1}) ne commence pas par 10 ou ne finit pas par 20 dans SessionStartEnd.")
        if df2.iloc[0]['SessionStartEnd'] != 10:
            print(f"Attention : Le deuxième fichier ({output_file_path2}) ne commence pas par 10 dans SessionStartEnd.")

elif option == 'a':
    check_session_start_end(session_start_end, time_stamp_opening)
elif option == 't':
    start_time = time.time()
    number_of_sessions = calculate_and_display_sessions(df)
    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Nombre de sessions de trading : {number_of_sessions}")
    print(f"Temps d'exécution de calculate_trading_sessions : {execution_time:.4f} secondes")
else:
    print("Option non reconnue. Veuillez appuyer sur 'c' ou 's' ou 'a'.")