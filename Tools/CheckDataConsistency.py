import csv
import sys
from colorama import init, Fore, Style

from colorama import init, Fore, Style


# Initialiser colorama pour Windows
init()

def calculate_sums(row, fields_blw, fields_abv):
    try:
        calc_vol_blw = sum(float(row[field]) for field in fields_blw)
        calc_vol_abv = sum(float(row[field]) for field in fields_abv)
        return calc_vol_blw, calc_vol_abv
    except KeyError as e:
        print(f"Erreur : Champ manquant - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Erreur : Valeur non numérique trouvée - {e}")
        sys.exit(1)

def format_comparison(label, value1, value2):
    if value1 != value2:
        return f"{label}: {Fore.RED}{value1} != {value2}{Style.RESET_ALL}"
    else:
        return f"{label}: {Fore.GREEN}{value1} == {value2}{Style.RESET_ALL}"

file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\\444_0_6TP_1SL_newBB\\of_raw_candles_dataNew.csv"

fields_blw = [
    "upTickVolBlwBidDesc", "downTickVolBlwBidDesc", "repeatUpTickVolBlwBidDesc",
    "repeatDownTickVolBlwBidDesc", "unknownTickVolBlwBidDesc", "upTickVolBlwAskDesc",
    "downTickVolBlwAskDesc", "repeatUpTickVolBlwAskDesc", "repeatDownTickVolBlwAskDesc",
    "unknownTickVolBlwAskDesc", "upTickVolBlwBidAsc", "downTickVolBlwBidAsc",
    "repeatUpTickVolBlwBidAsc", "repeatDownTickVolBlwBidAsc", "unknownTickVolBlwBidAsc",
    "upTickVolBlwAskAsc", "downTickVolBlwAskAsc", "repeatUpTickVolBlwAskAsc",
    "repeatDownTickVolBlwAskAsc", "unknownTickVolBlwAskAsc"
]

fields_abv = [
    "upTickVolAbvBidDesc", "downTickVolAbvBidDesc", "repeatUpTickVolAbvBidDesc",
    "repeatDownTickVolAbvBidDesc", "unknownTickVolAbvBidDesc", "upTickVolAbvAskDesc",
    "downTickVolAbvAskDesc", "repeatUpTickVolAbvAskDesc", "repeatDownTickVolAbvAskDesc",
    "unknownTickVolAbvAskDesc", "upTickVolAbvBidAsc", "downTickVolAbvBidAsc",
    "repeatUpTickVolAbvBidAsc", "repeatDownTickVolAbvBidAsc", "unknownTickVolAbvBidAsc",
    "upTickVolAbvAskAsc", "downTickVolAbvAskAsc", "repeatUpTickVolAbvAskAsc",
    "repeatDownTickVolAbvAskAsc", "unknownTickVolAbvAskAsc"
]

fields_6ticks_blw = [
    "upTickVol6TicksBlwBid", "downTickVol6TicksBlwBid", "repeatUpTickVol6TicksBlwBid",
    "repeatDownTickVol6TicksBlwBid", "unknownTickVol6TicksBlwBid", "upTickVol6TicksBlwAsk",
    "downTickVol6TicksBlwAsk", "repeatUpTickVol6TicksBlwAsk", "repeatDownTickVol6TicksBlwAsk",
    "unknownTickVol6TicksBlwAsk"
]

fields_6ticks_abv = [
    "upTickVol6TicksAbvBid", "downTickVol6TicksAbvBid", "repeatUpTickVol6TicksAbvBid",
    "repeatDownTickVol6TicksAbvBid", "unknownTickVol6TicksAbvBid", "upTickVol6TicksAbvAsk",
    "downTickVol6TicksAbvAsk", "repeatUpTickVol6TicksAbvAsk", "repeatDownTickVol6TicksAbvAsk",
    "unknownTickVol6TicksAbvAsk"
]

# Initialiser les compteurs
total_rows = 0
inconsistent_rows = 0
inconsistent_rows_trade_dir_1 = 0
inconsistent_rows_trade_dir_minus_1 = 0

try:
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            total_rows += 1

            # Lire les valeurs de la ligne
            vol_blw = float(row['VolBlw'])
            vol_abv = float(row['VolAbv'])
            delta_blw = float(row['DeltaBlw'])
            delta_abv = float(row['DeltaAbv'])
            volume = float(row['volume'])
            delta = float(row['delta'])
            vol_blw_6tick = float(row['VolBlw_6Tick'])
            vol_abv_6tick = float(row['VolAbv_6Tick'])

            # Convertir candle_dir et trade_dir en entier après les avoir transformés en float
            candle_dir = int(float(row['candleDir']))
            trade_dir = int(float(row['candleDir']))

            # Calculer les sommes des volumes
            calc_vol_blw, calc_vol_abv = calculate_sums(row, fields_blw, fields_abv)
            calc_volume = calc_vol_blw + calc_vol_abv
            calc_delta = delta_blw + delta_abv

            # Calculer les sommes pour VolBlw_6Tick et VolAbv_6Tick
            calc_vol_blw_6tick = sum(float(row[field]) for field in fields_6ticks_blw)
            calc_vol_abv_6tick = sum(float(row[field]) for field in fields_6ticks_abv)

            # Vérifier les incohérences et préparer les messages pour toutes les comparaisons
            has_error = False
            comparison_messages = []

            comparison_messages.append(format_comparison("VolBlw", vol_blw, calc_vol_blw))
            comparison_messages.append(format_comparison("VolAbv", vol_abv, calc_vol_abv))
            comparison_messages.append(format_comparison("Volume", volume, calc_volume))
            comparison_messages.append(format_comparison("Delta", delta, calc_delta))
            comparison_messages.append(format_comparison("VolBlw_6Tick", vol_blw_6tick, calc_vol_blw_6tick))
            comparison_messages.append(format_comparison("VolAbv_6Tick", vol_abv_6tick, calc_vol_abv_6tick))

            # Vérifier s'il y a au moins une erreur dans la ligne
            has_error = any(Fore.RED in msg for msg in comparison_messages)

            if has_error:
                inconsistent_rows += 1
                if trade_dir == 1:
                    inconsistent_rows_trade_dir_1 += 1
                elif trade_dir == -1:
                    inconsistent_rows_trade_dir_minus_1 += 1

                print(f"\nLigne {total_rows}:")
                for message in comparison_messages:
                    print(message)

    # Calculer le pourcentage d'incohérence
    if total_rows > 0:
        inconsistency_percentage = (inconsistent_rows / total_rows) * 100
        inconsistency_percentage_trade_dir_1 = (inconsistent_rows_trade_dir_1 / total_rows) * 100
        inconsistency_percentage_trade_dir_minus_1 = (inconsistent_rows_trade_dir_minus_1 / total_rows) * 100

        print(f"\nRésumé:")
        print(f"Nombre total de lignes : {total_rows}")
        print(f"Nombre total de lignes avec incohérence : {inconsistent_rows}")
        print(f"Pourcentage total d'incohérence : {inconsistency_percentage:.2f}%")
        print(f"Nombre de lignes avec incohérence où tradeDir = 1 : {inconsistent_rows_trade_dir_1}")
        print(f"Pourcentage d'incohérence où tradeDir = 1 : {inconsistency_percentage_trade_dir_1:.2f}%")
        print(f"Nombre de lignes avec incohérence où tradeDir = -1 : {inconsistent_rows_trade_dir_minus_1}")
        print(f"Pourcentage d'incohérence où tradeDir = -1 : {inconsistency_percentage_trade_dir_minus_1:.2f}%")
    else:
        print("Aucune ligne n'a été trouvée dans le fichier.")

except FileNotFoundError:
    print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
    sys.exit(1)
except csv.Error as e:
    print(f"Erreur lors de la lecture du fichier CSV : {e}")
    sys.exit(1)

# Afficher la valeur de la 2ème ligne et de la 1ère colonne, ainsi que celle de la dernière ligne et de la 1ère colonne
try:
    with open(file_path, 'r') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=';'))  # Lire toutes les lignes du fichier CSV

        if len(reader) >= 2:
            # Récupérer la valeur de la 2ème ligne et 1ère colonne
            value_2nd_line_1st_column = reader[1][0]
            print(f"\nValeur de la 2ème ligne et 1ère colonne : {value_2nd_line_1st_column}")

            # Récupérer la valeur de la dernière ligne et 1ère colonne
            value_last_line_1st_column = reader[-1][0]
            print(f"Valeur de la dernière ligne et 1ère colonne : {value_last_line_1st_column}")
        else:
            print("Le fichier ne contient pas assez de lignes pour afficher les valeurs demandées.")

except FileNotFoundError:
    print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
except csv.Error as e:
    print(f"Erreur lors de la lecture du fichier CSV : {e}")