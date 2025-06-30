import pandas as pd
import os
import datetime
from func_standard import *
import platform

# Paramètre pour utiliser les valeurs par défaut sans interaction utilisateur
USE_DEFAUT_PARAM_4_SPLIT_SESSION = False

if __name__ == "__main__":

    if platform.system() != "Darwin":
        directory_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra_chart\xTickReversal\simu\5_0_5TP_6SL_lundi soir\merge"
    else:
        directory_path = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"

    file_name = "Step5_5_0_5TP_6SL_010124_090625_extractOnlyFullSession_OnlyShort_feat.csv"
    file_path = os.path.join(directory_path, file_name)
    # Vérifier que le fichier existe
    if not os.path.isfile(file_path):
        print_notification(f"Erreur : Le fichier {file_path} n'existe pas")
    else:
        print_notification(f"Chargement du fichier {file_path}")
        df, encoding_used = load_csv_with_encoding(file_path)

        if df is None:
            print_notification("Impossible de charger le fichier avec les encodages disponibles. Fin du programme.")
        else:
            print_notification(f"Fichier debut du split du fichier Step 5 pour former Train, test, val1 et Val")
            # Appeler la fonction avec le DataFrame et les paramètres
            diviser_fichier_par_sessions(df, directory_path, file_name,
                                         use_default_params=USE_DEFAUT_PARAM_4_SPLIT_SESSION,
                                         encoding_used="ISO-8859-1") #ISO-8859-1