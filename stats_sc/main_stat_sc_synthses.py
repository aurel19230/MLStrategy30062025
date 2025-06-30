import os
import pandas as pd
import glob
import numpy as np
from collections import Counter


def process_stat_files(directory_path, output_filename="combined_unique_stats.csv", min_occurrences=2,
                       min_common_trades=20):
    """
    Lit tous les fichiers main_stat_sc* dans le répertoire spécifié,
    extrait les trades qui apparaissent dans plusieurs fichiers,
    et applique un système de vote unanime pour déterminer leur validité.

    Args:
        directory_path (str): Chemin du répertoire contenant les fichiers
        output_filename (str): Nom du fichier de sortie
        min_occurrences (int): Nombre minimum de fichiers dans lesquels un trade doit apparaître
        min_common_trades (int): Nombre minimum de trades en commun pour analyser une paire de fichiers
    """
    # Recherche de tous les fichiers main_stat_sc*
    pattern = os.path.join(directory_path, "main_stat_sc*.csv")
    stat_files = glob.glob(pattern)

    if not stat_files:
        print(f"Aucun fichier main_stat_sc*.csv trouvé dans {directory_path}")
        return None

    print(f"Nombre de fichiers trouvés: {len(stat_files)}")
    print("Fichiers:", stat_files)

    # Extraire juste les noms des fichiers sans le chemin
    file_basenames = [os.path.basename(f) for f in stat_files]

    # Colonnes pour déterminer les doublons
    indicator_columns = [
        'rsi_', 'macd', 'macd_signal', 'macd_hist',
        'timeElapsed2LastBar', 'timeStampOpening', 'ratio_deltaRevZone_VolCandle'
    ]

    # Dictionnaire pour suivre les occurrences de chaque trade unique
    trade_occurrences = {}
    trade_data = {}  # Pour stocker les données complètes de chaque trade
    trade_results = {}  # Pour stocker les résultats (win/loss) de chaque trade dans chaque fichier

    # Dictionnaire pour stocker les dataframes individuels
    dataframes_by_file = {}

    # Statistiques pour chaque fichier individuel
    file_stats = {}
    total_rows_before = 0

    # Structure pour stocker les trades uniques de chaque fichier
    file_unique_trades = {}

    # Lecture de tous les fichiers et identification des doublons inter-fichiers
    for file in stat_files:
        try:
            df = pd.read_csv(file, sep=';')
            file_basename = os.path.basename(file)
            dataframes_by_file[file_basename] = df
            file_stats[file_basename] = {'total_rows': len(df)}
            total_rows_before += len(df)

            # Vérifier que les colonnes nécessaires existent
            valid_indicator_columns = [col for col in indicator_columns if col in df.columns]

            if valid_indicator_columns:
                # Compter les doublons dans ce fichier
                duplicate_mask = df.duplicated(subset=valid_indicator_columns, keep='first')
                duplicates_count = duplicate_mask.sum()
                file_stats[file_basename]['duplicates_internal'] = duplicates_count
                file_stats[file_basename]['unique_rows'] = len(df) - duplicates_count

                # Stocker les trades uniques de ce fichier
                unique_df = df.drop_duplicates(subset=valid_indicator_columns)
                file_unique_trades[file_basename] = set()

                # Enregistrer chaque trade unique de ce fichier
                for _, row in unique_df.iterrows():
                    # Créer une clé unique basée sur les valeurs des colonnes d'indicateurs
                    key_values = tuple(row[col] for col in valid_indicator_columns if col in row)
                    file_unique_trades[file_basename].add(key_values)

                    if key_values not in trade_occurrences:
                        trade_occurrences[key_values] = []
                        trade_data[key_values] = row.to_dict()
                        trade_results[key_values] = {}

                    trade_occurrences[key_values].append(file_basename)

                    # Stocker le résultat du trade dans ce fichier
                    if 'trade_pnl' in row:
                        trade_results[key_values][file_basename] = row['trade_pnl'] > 0

                print(
                    f"Fichier {file_basename}: {len(df)} lignes, {duplicates_count} doublons internes, {len(df) - duplicates_count} uniques")
            else:
                print(f"Fichier {file_basename}: {len(df)} lignes (aucune colonne d'indicateur valide trouvée)")
                file_stats[file_basename]['duplicates_internal'] = 'N/A'
                file_stats[file_basename]['unique_rows'] = 'N/A'

        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")

    # Concaténer tous les dataframes
    all_data = pd.concat(dataframes_by_file.values(), ignore_index=True)

    # Vérification des données lues
    if all_data.empty:
        print("Aucune donnée n'a été lue depuis les fichiers")
        return None

    # S'assurer que nous utilisons les colonnes qui existent réellement
    valid_indicator_columns = [col for col in indicator_columns if col in all_data.columns]
    if not valid_indicator_columns:
        print(
            "Aucune colonne d'indicateur valide trouvée. Utilisation de toutes les colonnes pour détecter les doublons.")
        valid_indicator_columns = all_data.columns.tolist()
    else:
        print(f"Colonnes utilisées pour détecter les doublons: {valid_indicator_columns}")

    # Suppression des doublons basés sur les colonnes d'indicateurs techniques
    duplicate_mask = all_data.duplicated(subset=valid_indicator_columns, keep='first')
    duplicates_count = duplicate_mask.sum()
    unique_data = all_data[~duplicate_mask].copy()

    # Créer une matrice des doublons entre fichiers
    duplicate_matrix = pd.DataFrame(0, index=file_basenames, columns=file_basenames)

    # Remplir la matrice avec le nombre de doublons entre chaque paire de fichiers
    for file1 in file_basenames:
        for file2 in file_basenames:
            if file1 != file2:  # Ne pas compter les doublons d'un fichier avec lui-même
                if file1 in file_unique_trades and file2 in file_unique_trades:
                    # Compter les éléments communs entre les deux ensembles
                    duplicates = len(file_unique_trades[file1].intersection(file_unique_trades[file2]))
                    duplicate_matrix.loc[file1, file2] = duplicates

    # Afficher la matrice des doublons avec une meilleure présentation
    pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
    pd.set_option('display.width', 1000)  # Augmenter la largeur d'affichage

    print("\n=== MATRICE DES DOUBLONS ENTRE FICHIERS ===")
    print("Cette matrice montre le nombre de trades en commun entre chaque paire de fichiers:")
    print(duplicate_matrix)

    # Créer une version plus lisible avec des noms de fichiers courts
    short_names = {name: f"sc{i}" for i, name in enumerate(file_basenames)}
    readable_matrix = duplicate_matrix.copy()
    readable_matrix.index = [short_names[name] for name in readable_matrix.index]
    readable_matrix.columns = [short_names[name] for name in readable_matrix.columns]

    print("\n=== MATRICE DES DOUBLONS (FORMAT SIMPLIFIÉ) ===")
    print("Légende:")
    for name, short in short_names.items():
        print(f"  {short} = {name}")
    print("\nMatrice:")
    print(readable_matrix)

    # Distribution du nombre de fichiers où apparaît chaque trade
    occurrence_counts = Counter([len(v) for v in trade_occurrences.values()])
    print("\nDistribution des occurrences:")
    for count, num_trades in sorted(occurrence_counts.items()):
        print(f"  Trades apparaissant dans {count} fichier(s): {num_trades}")

    print(f"\nTotal des lignes lues: {len(all_data)}")
    print(f"Nombre de lignes uniques: {len(unique_data)}")
    print(f"Nombre de doublons supprimés: {duplicates_count}")

    # Statistiques globales
    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    total_gains = 0
    total_losses = 0

    if 'trade_pnl' in unique_data.columns:
        # Marquer les trades gagnants/perdants
        unique_data['is_winning'] = unique_data['trade_pnl'] > 0

        # Compter les trades
        winning_trades = unique_data['is_winning'].sum()
        losing_trades = len(unique_data) - winning_trades

        # Calculer les PnL
        total_pnl = unique_data['trade_pnl'].sum()
        total_gains = unique_data.loc[unique_data['is_winning'], 'trade_pnl'].sum()
        total_losses = unique_data.loc[~unique_data['is_winning'], 'trade_pnl'].sum()

        # Calculer les moyennes
        avg_win = unique_data.loc[unique_data['is_winning'], 'trade_pnl'].mean()
        avg_loss = unique_data.loc[~unique_data['is_winning'], 'trade_pnl'].mean()

        # Calculer le ratio risque/récompense
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Calculer l'expectancy
        winrate = winning_trades / len(unique_data) * 100 if len(unique_data) > 0 else 0
        expectancy = (winrate / 100 * avg_win) + ((100 - winrate) / 100 * avg_loss)

        print("\n=== RÉSUMÉ DES TRADES APRÈS DÉDUPLICATION ===")
        print(f"Trades totaux avant déduplication: {total_rows_before}")
        print(
            f"Trades conservés après déduplication: {len(unique_data)} ({len(unique_data) / total_rows_before * 100:.2f}%)")
        print(f"Trades réussis conservés: {winning_trades}")
        print(f"Trades échoués conservés: {losing_trades}")
        print(f"Winrate: {winrate:.2f}%")
        print(f"PnL total: {total_pnl:.2f}")
        print(f"Gains totaux: {total_gains:.2f}")
        print(f"Pertes totales: {total_losses:.2f}")
        print(f"Gain moyen: {avg_win:.2f}")
        print(f"Perte moyenne: {avg_loss:.2f}")
        print(f"Ratio risque/récompense: {reward_risk_ratio:.2f}")
        print(f"Expectancy par trade: {expectancy:.2f}")

    # Filtrer les trades qui apparaissent dans au moins min_occurrences fichiers
    multi_occurrence_trades = {k: v for k, v in trade_occurrences.items() if len(v) >= min_occurrences}

    # Pour chaque nombre d'occurrences possible, analyser les trades correspondants
    occurrences_stats = {}

    for occ_count in range(2, max([len(v) for v in trade_occurrences.values()]) + 1):
        trades_with_occ = {k: v for k, v in trade_occurrences.items() if len(v) == occ_count}

        if not trades_with_occ:
            continue

        # Analyse des trades qui apparaissent dans exactement occ_count fichiers
        winning_trades = []  # Liste des trades gagnants (pour calculer le PnL)
        losing_trades = []  # Liste des trades perdants (pour calculer le PnL)

        for key, files in trades_with_occ.items():
            # Un trade est considéré comme valide uniquement si TOUS les algorithmes le marquent comme gagnant
            if all(trade_results.get(key, {}).get(file, False) for file in files):
                winning_trades.append(key)
            else:
                losing_trades.append(key)

        total_trades = len(winning_trades) + len(losing_trades)
        winrate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Calculer le PnL total pour les trades gagnants et perdants
        winning_pnl = sum(trade_data.get(key, {}).get('trade_pnl', 0) for key in winning_trades)
        losing_pnl = sum(trade_data.get(key, {}).get('trade_pnl', 0) for key in losing_trades)
        total_pnl = winning_pnl + losing_pnl

        occurrences_stats[occ_count] = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'winrate': winrate,
            'winning_pnl': winning_pnl,
            'losing_pnl': losing_pnl,
            'total_pnl': total_pnl
        }

        print(f"\n=== ANALYSE DES TRADES APPARAISSANT DANS {occ_count} FICHIERS ===")
        print(f"Nombre total de trades: {total_trades}")
        print(f"Trades unanimement gagnants: {len(winning_trades)}")
        print(f"Trades non unanimes (au moins un échec): {len(losing_trades)}")
        print(f"Winrate (trades unanimement gagnants): {winrate:.2f}%")
        print(f"PnL total: {total_pnl:.2f}")
        print(f"PnL des trades gagnants: {winning_pnl:.2f}")
        print(f"PnL des trades perdants: {losing_pnl:.2f}")

    # Analyse des paires de fichiers ayant plus de min_common_trades trades en commun
    print("\n=== ANALYSE DES PAIRES AVEC PLUS DE", min_common_trades, "TRADES EN COMMUN ===")

    # Éviter les paires redondantes (comme sc5-sc4 et sc4-sc5)
    analyzed_pairs = set()
    significant_pairs = []

    for i, file1 in enumerate(file_basenames):
        for j, file2 in enumerate(file_basenames[i + 1:], i + 1):  # Ne traiter que les paires où file1 < file2
            if duplicate_matrix.loc[file1, file2] >= min_common_trades:
                significant_pairs.append((file1, file2))
                analyzed_pairs.add((file1, file2))

    pairs_stats = {}

    for file1, file2 in significant_pairs:
        # Trouver les trades communs aux deux fichiers
        common_trades = file_unique_trades[file1].intersection(file_unique_trades[file2])

        # Analyser ces trades
        winning_both = 0
        winning_file1_only = 0
        winning_file2_only = 0
        losing_both = 0

        total_pnl = 0
        unanimous_pnl = 0

        for key in common_trades:
            result_file1 = trade_results.get(key, {}).get(file1, False)
            result_file2 = trade_results.get(key, {}).get(file2, False)

            # Compter les cas selon le résultat
            if result_file1 and result_file2:
                winning_both += 1
            elif result_file1 and not result_file2:
                winning_file1_only += 1
            elif not result_file1 and result_file2:
                winning_file2_only += 1
            else:
                losing_both += 1

            # Calculer le PnL
            if key in trade_data and 'trade_pnl' in trade_data[key]:
                pnl = trade_data[key]['trade_pnl']
                total_pnl += pnl

                # Ajouter au PnL unanime si les deux fichiers sont d'accord sur le résultat
                if result_file1 == result_file2:
                    unanimous_pnl += pnl

        # Calculer l'accord entre les algorithmes
        total_common = len(common_trades)
        agreement_rate = (winning_both + losing_both) / total_common * 100 if total_common > 0 else 0

        # Stocker les statistiques
        pair_name = f"{short_names[file1]}-{short_names[file2]}"
        pairs_stats[pair_name] = {
            'file1': file1,
            'file2': file2,
            'common_trades': total_common,
            'winning_both': winning_both,
            'winning_file1_only': winning_file1_only,
            'winning_file2_only': winning_file2_only,
            'losing_both': losing_both,
            'agreement_rate': agreement_rate,
            'total_pnl': total_pnl,
            'unanimous_pnl': unanimous_pnl
        }

        print(f"\nPaire {pair_name} ({file1} - {file2}):")
        print(f"Trades communs: {total_common}")
        print(f"Trades gagnants dans les deux: {winning_both}")
        print(f"Trades gagnants dans {short_names[file1]} seulement: {winning_file1_only}")
        print(f"Trades gagnants dans {short_names[file2]} seulement: {winning_file2_only}")
        print(f"Trades perdants dans les deux: {losing_both}")
        print(f"Taux d'accord: {agreement_rate:.2f}%")
        print(f"PnL total: {total_pnl:.2f}")
        print(f"PnL des trades avec accord unanime: {unanimous_pnl:.2f}")

    # Créer un DataFrame avec tous les trades multi-occurrences
    all_multi_occ_data = []

    for occ_count in range(2, max([len(v) for v in trade_occurrences.values()]) + 1):
        trades_with_occ = {k: v for k, v in trade_occurrences.items() if len(v) == occ_count}

        for key, files in trades_with_occ.items():
            # Récupérer les données de ce trade
            if key in trade_data:
                trade_info = trade_data[key].copy()

                # Vérifier si tous les algorithmes considèrent ce trade comme gagnant
                is_unanimous_win = all(trade_results.get(key, {}).get(file, False) for file in files)

                # Ajouter des informations supplémentaires
                trade_info['occurrence_count'] = occ_count
                trade_info['appears_in_files'] = ', '.join(files)
                trade_info['unanimous_win'] = is_unanimous_win

                # Pour chaque fichier, ajouter si ce trade est gagnant dans ce fichier
                for file in file_basenames:
                    if file in files:
                        trade_info[f'win_in_{short_names[file]}'] = trade_results.get(key, {}).get(file, False)
                    else:
                        trade_info[f'win_in_{short_names[file]}'] = None

                all_multi_occ_data.append(trade_info)

    # Créer un DataFrame avec tous les trades multi-occurrences
    if all_multi_occ_data:
        all_multi_occ_df = pd.DataFrame(all_multi_occ_data)
        all_multi_occ_path = os.path.join(directory_path, "all_multi_occurrence_trades.csv")
        all_multi_occ_df.to_csv(all_multi_occ_path, sep=';', index=False)
        print(f"\n✓ Fichier de tous les trades multi-occurrences enregistré: {all_multi_occ_path}")

    # Créer un DataFrame uniquement avec les trades unanimement gagnants
    unanimous_win_data = [data for data in all_multi_occ_data if data.get('unanimous_win', False)]

    if unanimous_win_data:
        unanimous_win_df = pd.DataFrame(unanimous_win_data)
        unanimous_win_path = os.path.join(directory_path, "unanimous_winning_trades.csv")
        unanimous_win_df.to_csv(unanimous_win_path, sep=';', index=False)
        print(f"✓ Fichier des trades unanimement gagnants enregistré: {unanimous_win_path}")

    # Enregistrer les statistiques par nombre d'occurrences
    occurrences_stats_df = pd.DataFrame.from_dict(occurrences_stats, orient='index')
    occurrences_stats_path = os.path.join(directory_path, "occurrences_statistics.csv")
    occurrences_stats_df.to_csv(occurrences_stats_path, sep=';')
    print(f"✓ Fichier des statistiques par nombre d'occurrences enregistré: {occurrences_stats_path}")

    # Enregistrer les statistiques des paires significatives
    if pairs_stats:
        pairs_stats_df = pd.DataFrame.from_dict(pairs_stats, orient='index')
        pairs_stats_path = os.path.join(directory_path, "significant_pairs_statistics.csv")
        pairs_stats_df.to_csv(pairs_stats_path, sep=';')
        print(f"✓ Fichier des statistiques des paires significatives enregistré: {pairs_stats_path}")

    # Enregistrer la matrice dans un fichier CSV
    matrix_path = os.path.join(directory_path, "duplicate_matrix.csv")
    duplicate_matrix.to_csv(matrix_path, sep=';')
    print(f"✓ Matrice des doublons enregistrée: {matrix_path}")

    # Enregistrer également la version simplifiée
    simple_matrix_path = os.path.join(directory_path, "duplicate_matrix_simple.csv")
    readable_matrix.to_csv(simple_matrix_path, sep=';')
    print(f"✓ Matrice simplifiée des doublons enregistrée: {simple_matrix_path}")

    return {
        'occurrences_stats': occurrences_stats,
        'pairs_stats': pairs_stats
    }


# Exemple d'utilisation:
if __name__ == "__main__":
    # Remplacez par votre chemin de répertoire
    import platform
    if platform.system() != "Darwin":
        DIRECTORY_PATH = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\5_0_5TP_1SL_1\\\merge"

    else:
        DIRECTORY_PATH = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"
    process_stat_files(DIRECTORY_PATH, min_occurrences=2, min_common_trades=20)

