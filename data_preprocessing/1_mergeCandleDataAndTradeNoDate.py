from func_standard import convert_to_unix_timestamp, convert_from_unix_timestamp, timestamp_to_date_utc, \
    date_to_timestamp_utc, print_notification
import os
import csv
from datetime import datetime
import numpy as np
from numba import njit, prange

#si les colonnes sont modifiées:
index_SessionStartEnd=5
index_deltaTimestampOpening=3
index_indexTrade_result = 6


# Répertoires
directory = (
    "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra_chart/xTickReversal/simu/5_0_5TP_6SL/")
@njit(parallel=True)
def apply_20_10_logic_numba(data):
    for i in prange(index_SessionStartEnd, len(data)):
        if int(data[i, index_SessionStartEnd]) == 10:
            data[i - 1, index_SessionStartEnd] = 20
        if (data[i, index_deltaTimestampOpening] < 300 and data[i - 1, index_deltaTimestampOpening] > 600) and (int(data[i, index_SessionStartEnd]) != 10 and int(data[i - 1, index_SessionStartEnd]) != 20):
            data[i - 1, index_SessionStartEnd] = 20
            data[i, index_SessionStartEnd] = 10
    return data


# def apply_20_10_logic_noNumba(data):
#     for i in range(4, len(data)):
#         if int(data[i][4]) == 10:
#             data[i - 1][4] = 20
#         if (data[i][2] < 300 and data[i - 1][2] > 600) and (int(data[i][4]) != 10 and int(data[i - 1][4]) != 20):
#             print(f"Valeur à {data[i - 1][0]} : {data[i][0]}")
#             user_input = input("Voulez-vous ajouter un 20 à data[i-1][4] et un 10 à data[i][4]? (y/n): ")
#             if user_input.lower() == 'y':
#                 data[i - 1][4] = 20
#                 data[i][4] = 10
#     return data


@njit(parallel=True)
def process_data_numba(candles_data, trades_data):
    #result = np.empty((len(candles_data), len(candles_data[0]) + 3), dtype=np.float64)
    result = np.empty((len(candles_data), len(candles_data[0]) + 7), dtype=np.float64)
    for i in prange(len(candles_data)):
        index = int(candles_data[i, index_indexTrade_result])
        match = trades_data[trades_data[:, 0] == index]
        if len(match) > 0:
            trade_result = int(match[0, 1])
            trade_pnl = float(match[0, 2])
            tp1_pnl = float(match[0, 3])
            tp2_pnl = float(match[0, 4])
            tp3_pnl = float(match[0, 5])
            sl_pnl = float(match[0, 6])
            if trade_result == 2:
                trade_dir, trade_res = -1, 1
            elif trade_result == 1:
                trade_dir, trade_res = 1, 1
            elif trade_result == -1:
                trade_dir, trade_res = 1, -1
            elif trade_result == -2:
                trade_dir, trade_res = -1, -1
            else:
                trade_dir, trade_res = 0, 0
        else:
            trade_dir, trade_res, trade_pnl, tp1_pnl, tp2_pnl, tp3_pnl, sl_pnl = 0, 99, 0.0, 0.0, 0.0, 0.0, 0.0

        #result[i] = np.concatenate((candles_data[i], np.array([trade_dir, trade_res, trade_pnl], dtype=np.float64)))
        result[i] = np.concatenate((candles_data[i], np.array([trade_dir, trade_res, trade_pnl,tp1_pnl,
                                                               tp2_pnl,tp3_pnl,sl_pnl], dtype=np.float64)))

    result = apply_20_10_logic_numba(result)
    return result


# def process_data_with_date(candles_data, trades_data):
#     result = []
#
#     for i in range(len(candles_data)):
#         index = int(candles_data[i, 4])
#         match = trades_data[trades_data[:, 0] == index]
#         if len(match) > 0:
#             trade_result = int(match[0, 1])
#             trade_pnl = float(match[0, 2])
#             if trade_result == 2:
#                 trade_dir, trade_res = -1, 1
#             elif trade_result == 1:
#                 trade_dir, trade_res = 1, 1
#             elif trade_result == -1:
#                 trade_dir, trade_res = 1, -1
#             elif trade_result == -2:
#                 trade_dir, trade_res = -1, -1
#             else:
#                 trade_dir, trade_res = 0, 0
#         else:
#             trade_dir, trade_res, trade_pnl = 0, 99, 0.0
#
#         timestamp = int(candles_data[i, 0])
#         formatted_date = timestamp_to_date_utc(timestamp)
#         row = [formatted_date] + candles_data[i].tolist() + [trade_dir, trade_res, trade_pnl]
#         result.append(row)
#
#     result = apply_20_10_logic_noNumba(result)
#     return result

import glob

def process_single_file(directory, directoryMerge, suffix, add_date_column, delimiter=";"):
    """
    Traite un seul ensemble de fichiers avec le suffixe donné
    """
    withDate = "_withDate" if add_date_column.lower() == 'd' else ""

    # Recherche des fichiers se terminant par _3.csv ou _4.csv etc.
    files_pattern = os.path.join(directory, f"*_{suffix}.csv")

    # Trouver les fichiers correspondants
    matching_files = glob.glob(files_pattern)

    if not matching_files:
        print_notification(f"Les fichiers se terminant par _{suffix} n'existent pas. Passage au suivant.")
        print(f"Pattern recherché: {files_pattern}")
        return False

    print(f"\nFichiers trouvés avec le suffixe _{suffix}:")
    for file in matching_files:
        print(f"- {os.path.basename(file)}")

    # Identifier les fichiers candles et trades
    candles_file = None
    trades_file = None

    for file in matching_files:
        filename = os.path.basename(file)
        if 'candles' in filename.lower():
            candles_file = file
        elif 'trade' in filename.lower():
            trades_file = file

    if not candles_file or not trades_file:
        print_notification("Impossible de trouver un fichier candles et un fichier trades correspondants.")
        return False

    # Lire les données
    print_notification(f"Lecture des données de {os.path.basename(candles_file)}")
    with open(candles_file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)
        candles_data = np.array([row for row in reader], dtype=float)

    print_notification(f"Lecture des données de {os.path.basename(trades_file)}")
    trades_data = np.genfromtxt(trades_file, delimiter=delimiter, skip_header=1)

    # Extraire les dates
    start_timestamp, end_timestamp = int(candles_data[0, 0]), int(candles_data[-1, 0])
    start_date = datetime.utcfromtimestamp(start_timestamp).strftime('%d%m%y')
    end_date = datetime.utcfromtimestamp(end_timestamp).strftime('%d%m%y')

    # Nom du fichier de sortie
    output_file = f"Step1_{start_date}_{end_date}_4TicksRev_{suffix}{withDate}.csv"

    # # Traiter les données
    # if add_date_column.lower() == 'd':
    #     print_notification("Traitement des données avec ajout de la colonne 'date'")
    #     merged_data = process_data_with_date(candles_data, trades_data)
    # else:
    print_notification("Traitement des données sans ajout de la colonne 'date'")
    merged_data = process_data_numba(candles_data, trades_data)

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(directoryMerge, exist_ok=True)

    # Écrire les données fusionnées
    print_notification(f"Écriture des données fusionnées dans le fichier : {output_file}")
    with open(os.path.join(directoryMerge, output_file), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=delimiter)
        if add_date_column.lower() == 'd':
            headers = ['dateUTC'] + headers
        headers.extend(['tradeDir', 'tradeResult', 'trade_pnl', 'tp1_pnl', 'tp2_pnl', 'tp3_pnl', 'sl_pnl'])
        #headers.extend(['tradeDir', 'tradeResult', 'trade_pnl'])
        writer.writerow(headers)
        writer.writerows(merged_data)

    print_notification(f"Fusion des fichiers terminée pour le suffixe _{suffix}")
    return True


def main():
    # Demander à l'utilisateur s'il souhaite ajouter la colonne "date"
    add_date_column = input(
        "Voulez-vous ajouter une colonne 'date' ? (Appuyez sur Entrée pour non, ou tapez 'd' pour oui) : ")


    if os.path.exists(directory):
        if os.path.isdir(directory):
            print(f"Le dossier {directory} existe.")
        else:
            print(f"{directory} existe mais n'est pas un dossier.")
            return
    else:
        print(f"Le dossier {directory} n'existe pas.")
        return

    directoryMerge = os.path.join(directory, "merge")
    # Créer le répertoire merge s'il n'existe pas
    if not os.path.exists(directoryMerge):
        os.makedirs(directoryMerge)
        print_notification(f"Création du dossier merge: {directoryMerge}")

    while True:
        start_input = input("Entrez le numéro de suffixe de départ (ex: 3 pour _3) ou appuyez sur Entrée pour 1 : ")
        try:
            start_suffix = int(start_input) if start_input.strip() else 1
            break
        except ValueError:
            print("Veuillez entrer un nombre valide ou appuyez sur Entrée pour la valeur par défaut.")

    while True:
        end_input = input(
            f"Entrez le numéro de suffixe de fin (ex: 5 pour _5) ou appuyez sur Entrée pour {start_suffix} : ")
        try:
            end_suffix = int(end_input) if end_input.strip() else start_suffix
            if end_suffix >= start_suffix:
                break
            else:
                print("Le numéro de fin doit être supérieur ou égal au numéro de début.")
        except ValueError:
            print("Veuillez entrer un nombre valide ou appuyez sur Entrée pour la valeur par défaut.")

    print(f"\nTraitement des fichiers du suffixe _{start_suffix} au suffixe _{end_suffix}")

    # Traiter chaque fichier dans la plage
    for suffix_num in range(start_suffix, end_suffix + 1):
        suffix = str(suffix_num)  # Ne pas ajouter de underscore ici
        print_notification(f"\nTraitement des fichiers avec le suffixe _{suffix}")
        success = process_single_file(directory, directoryMerge, suffix, add_date_column)
        if not success:
            print_notification(f"Impossible de traiter les fichiers avec le suffixe _{suffix}")

    print_notification("Traitement de tous les fichiers terminé.")


if __name__ == "__main__":
    main()