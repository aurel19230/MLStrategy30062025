import pandas as pd
import os
import re
from func_standard import print_notification, load_data, calculate_naked_poc_distances, CUSTOM_SESSIONS, \
    save_features_with_sessions, remplace_0_nan_reg_slope_p_2d, process_reg_slope_replacement


def configurer_pandas():
    """Configure les options d'affichage de pandas."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.precision', 10)
    pd.set_option('display.float_format', lambda x: '%.10f' % x if abs(x) < 1e10 else '%.0f' % x)


def generate_output_filename(files, xtickRev_config_dir):
    """Génère le nom du fichier de sortie basé sur les fichiers d'entrée."""
    if not files:
        raise ValueError("La liste des fichiers est vide.")

    # Trier les fichiers
    sorted_files = sorted(files)

    # Trouver le fichier qui se termine par "_0.csv"
    first_file = next((f for f in sorted_files if f.endswith('_0.csv')), None)
    if first_file is None:
        raise ValueError("Aucun fichier ne se termine par '_0.csv'.")

    # Extraire la date du début à partir du fichier *0.csv
    start_date = first_file.split('_')[1]

    # Trouver le fichier avec le plus grand X dans *X.csv
    last_file = max(sorted_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Extraire la date de fin à partir du dernier fichier
    end_date = last_file.split('_')[2]

    # Générer le nom du fichier de sortie
    output_filename = f"Step2_{xtickRev_config_dir}_{start_date}_{end_date}.csv"
    return output_filename


def fix_trade_results_and_add_pnl(df, sl_value=-227, tp_value=175, output_suffix="_bugFixTradeResult1"):
    """
    Corrige les tradeResult (corrigé dans SC dans version 2) et ajoute des colonnes de PnL théorique.

    Args:
        df: DataFrame contenant les données
        sl_value: Valeur du Stop Loss théorique (négatif)
        tp_value: Valeur du Take Profit théorique (positif)
        output_suffix: Suffixe à ajouter au nom du fichier de sortie

    Returns:
        DataFrame modifié avec les nouvelles colonnes
    """
    print_notification("Correction des tradeResult et ajout des colonnes PnL théorique")

    # Faire une copie pour ne pas modifier le DataFrame original
    df_modified = df.copy()

    # 1. Lorsque sl_pnl > 0, mettre tradeResult à -1
    mask_sl_positive = df_modified['sl_pnl'] > 0
    if mask_sl_positive.any():
        print(f"  - {mask_sl_positive.sum()} lignes avec sl_pnl > 0 corrigées (tradeResult = -1)")
        df_modified.loc[mask_sl_positive, 'tradeResult'] = -1

    # 2. Ajouter les colonnes de PnL théorique
    df_modified['trade_pnl_theoric'] = 0
    df_modified.loc[df_modified['tradeResult'] == 1, 'trade_pnl_theoric'] = tp_value
    df_modified.loc[df_modified['tradeResult'] == -1, 'trade_pnl_theoric'] = sl_value

    df_modified['tp1_pnl_theoric'] = 0
    df_modified.loc[df_modified['tradeResult'] == 1, 'tp1_pnl_theoric'] = tp_value

    df_modified['sl_pnl_theoric'] = 0
    df_modified.loc[df_modified['tradeResult'] == -1, 'sl_pnl_theoric'] = sl_value

    print(f"  - Colonnes de PnL théorique ajoutées avec SL={sl_value} et TP={tp_value}")

    return df_modified


def main():
    # Définir les valeurs de PnL théorique
    SL_VALUE = -227
    TP_VALUE = 175

    # Chemin du répertoire
    directory_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\merge"
    file_name = "Step2_5_0_5TP_1SL_020924_011024.csv"

    # Construction du chemin complet du fichier
    file_path = os.path.join(directory_path, file_name)

    # Configuration de pandas
    configurer_pandas()

    try:
        # Charger les données
        print_notification(f"Chargement du fichier {file_name}")
        df = load_data(file_path)
        print(f"  - {len(df)} lignes chargées")

        # Correction des tradeResult et ajout des colonnes de PnL théorique
        df_modified = fix_trade_results_and_add_pnl(df, SL_VALUE, TP_VALUE)

        # Générer le nom du fichier de sortie
        output_file_name = file_name.replace('.csv', '_bugFixTradeResult1.csv')
        output_file_path = os.path.join(directory_path, output_file_name)

        # Sauvegarder le fichier
        print_notification(f"Sauvegarde du fichier {output_file_name}")
        df_modified.to_csv(output_file_path, sep=';', index=False)
        print(f"  - Fichier sauvegardé avec succès: {output_file_path}")

        print(f"\nTraitement terminé avec succès !")
        print(f"Nombre total de lignes dans le fichier final : {len(df_modified)}")

    except Exception as e:
        print(f"\nUne erreur s'est produite : {str(e)}")


if __name__ == "__main__":
    main()