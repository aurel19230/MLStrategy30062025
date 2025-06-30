import pandas as pd
import os
import time

# Configuration pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.0f' % x)  # Pour les nombres flottants

separator = ';'

# Définition des chemins et noms de fichiers
files_info = [
    {
        'path': r'C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\mai_spare\merge',
        'filename': 'Step1_180422_070622_4TicksRev_2',
        'role': 'source'
    },
    {
        'path': r'C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\merge',
        'filename': 'Step1_050522_111223_4TicksRev_Cleaned_2',
        'role': 'target'
    }
]

# Période d'analyse

start_timestamp = 1651788000
end_timestamp = 1654581207

#start_timestamp = 1702335600
#end_timestamp = 1706201125


def timestamp_to_date_utc_(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))


def load_and_filter_data(file_path, file_name):
    """Charge et filtre les données d'un fichier."""
    try:
        full_path = os.path.join(file_path, f'{file_name}.csv')
        data = pd.read_csv(full_path, sep=separator)

        # Remplacer les valeurs NaN par 0 avant la conversion en entier
        if 'timeStampOpening' in data.columns:
            data['timeStampOpening'] = data['timeStampOpening'].fillna(0).astype(int)

        return data
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_name}: {str(e)}")
        return None


def analyze_specific_data(data, start_ts=None, end_ts=None, title_prefix="", is_complete_analysis=False):
    """Analyse un ensemble de données spécifique"""
    if is_complete_analysis:
        # Pour l'analyse complète, prendre le premier et dernier timestamp du fichier
        file_start_ts = data['timeStampOpening'].iloc[0]
        file_end_ts = data['timeStampOpening'].iloc[-1]
        date_info = f"\nPériode complète du fichier:"
        date_info += f"\nDu timestamp {file_start_ts} au timestamp {file_end_ts}"
        date_info += f"\nDu {timestamp_to_date_utc_(file_start_ts)} au {timestamp_to_date_utc_(file_end_ts)}"
        filtered_data = data
    elif start_ts is not None and end_ts is not None:
        mask = (data['timeStampOpening'] >= start_ts) & (data['timeStampOpening'] <= end_ts)
        filtered_data = data[mask]
        date_info = f"\nPériode analysée: {start_ts} à {end_ts}"
        date_info += f"\nDates: {timestamp_to_date_utc_(start_ts)} à {timestamp_to_date_utc_(end_ts)}"

    print(f"{title_prefix}{date_info}")

    if not filtered_data.empty:
        count = filtered_data.shape[0]
        print(f"\nNombre total de lignes: {count}")

        tradeResult_counts = filtered_data['tradeResult'].value_counts().sort_index()
        print("\nDénombrement des résultats de trades:")
        print(tradeResult_counts)

        if 1 in tradeResult_counts.index and -1 in tradeResult_counts.index:
            ratio_1_minus1 = tradeResult_counts[1] / tradeResult_counts[-1]
            print(f"\nRatio (1/-1): {ratio_1_minus1:.2f}")

        total_trades = tradeResult_counts.sum()
        percentages = (tradeResult_counts / total_trades * 100).round(2)
        print("\nPourcentages des résultats:")
        for idx, percentage in percentages.items():
            print(f"Valeur {idx}: {percentage}%")

        trades_excl_99 = (tradeResult_counts[-1] + tradeResult_counts[1]
                          if (1 in tradeResult_counts.index and -1 in tradeResult_counts.index)
                          else 0)
        if trades_excl_99 > 0:
            win_rate = (tradeResult_counts[1] / trades_excl_99 * 100)
            print(f"\nWin Rate (excluant les 99): {win_rate:.2f}%")


def analyze_data():
    """Analyse les données des fichiers."""
    # Analyse des fichiers avec période spécifiée
    for file_info in files_info:
        print(f"\n{'='*80}")
        title_prefix = f"Analyse du fichier: {file_info['filename']}\n"
        title_prefix += f"Chemin: {file_info['path']}"
        print(title_prefix)
        print('='*80)

        data = load_and_filter_data(file_info['path'], file_info['filename'])
        if data is not None:
            analyze_specific_data(data, start_timestamp, end_timestamp, title_prefix)

    # Analyse complète du fichier target
    target_info = next(f for f in files_info if f['role'] == 'target')
    target_data = load_and_filter_data(target_info['path'], target_info['filename'])
    if target_data is not None:
        print(f"\n{'='*80}")
        title_prefix = f"Analyse COMPLÈTE du fichier TARGET: {target_info['filename']}\n"
        title_prefix += f"Chemin: {target_info['path']}"
        print(title_prefix)
        print('='*80)
        analyze_specific_data(target_data, None, None, title_prefix, is_complete_analysis=True)

def replace_data():
    """Remplace les données dans la période spécifiée."""
    print("\nDébut du processus de remplacement...")

    # Charger les fichiers source et cible
    print("Chargement des fichiers...")
    source_data = load_and_filter_data(files_info[0]['path'], files_info[0]['filename'])
    target_data = load_and_filter_data(files_info[1]['path'], files_info[1]['filename'])

    if source_data is None or target_data is None:
        return

    try:
        # Créer les masques pour la période spécifiée
        source_mask = (source_data['timeStampOpening'] >= start_timestamp) & (
                    source_data['timeStampOpening'] <= end_timestamp)
        target_mask = (target_data['timeStampOpening'] >= start_timestamp) & (
                    target_data['timeStampOpening'] <= end_timestamp)

        # Vérifier que les données source et cible ont les mêmes colonnes
        source_filtered = source_data[source_mask]
        target_filtered = target_data[target_mask]

        print(f"Nombre de lignes à remplacer: {len(target_filtered)}")
        print(f"Nombre de lignes de remplacement: {len(source_filtered)}")

        # Créer une copie du fichier cible
        modified_data = target_data.copy()

        # S'assurer que les types de données sont cohérents
        for column in modified_data.columns:
            if column in source_data.columns:
                # Gérer les types de données spéciaux
                if modified_data[column].dtype == 'int64':
                    source_data[column] = source_data[column].fillna(0).astype('int64')
                elif modified_data[column].dtype == 'float64':
                    source_data[column] = source_data[column].fillna(0.0).astype('float64')

        # Remplacer les données dans la période spécifiée
        modified_data.loc[target_mask] = source_data[source_mask].values

        # Générer le nom du fichier de sortie
        output_filename = files_info[1]['filename'] + 'Cleaned.csv'
        output_path = os.path.join(files_info[1]['path'], output_filename)

        # Sauvegarder le fichier modifié
        modified_data.to_csv(output_path, sep=separator, index=False)

        print(f"\nRemplacement effectué avec succès!")
        print(f"Fichier sauvegardé: {output_path}")

        # Afficher quelques statistiques sur le remplacement
        replaced_rows = sum(target_mask)
        print(f"\nStatistiques du remplacement:")
        print(f"Nombre de lignes remplacées: {replaced_rows}")
        print(f"Période de remplacement: de {timestamp_to_date_utc_(start_timestamp)}")
        print(f"                         à {timestamp_to_date_utc_(end_timestamp)}")

        # Vérifier l'intégrité des données après remplacement
        verification_data = pd.read_csv(output_path, sep=separator)
        print("\nVérification du fichier sauvegardé:")
        print(f"Nombre total de lignes: {len(verification_data)}")
        print("Types des colonnes après sauvegarde:")
        print(verification_data.dtypes)

    except Exception as e:
        print(f"Erreur lors du remplacement des données: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Fonction principale."""
    print("\n=== Programme d'analyse et de remplacement de données ===")
    print(f"Période d'analyse configurée: ")
    print(f"Du : {timestamp_to_date_utc_(start_timestamp)}")
    print(f"Au : {timestamp_to_date_utc_(end_timestamp)}")

    # Analyse initiale
    print("\nAnalyse initiale des fichiers:")
    analyze_data()

    while True:
        print("\n" + "=" * 50)
        print("Options disponibles:")
        print("1. Effectuer le remplacement des données")
        print("2. Voir l'analyse actuelle")
        print("3. Quitter le programme")
        choice = input("\nEntrez votre choix (1, 2 ou 3): ")

        if choice == '1':
            print("\nVous avez choisi d'effectuer le remplacement.")
            confirmation = input("Êtes-vous sûr de vouloir procéder au remplacement ? (o/n): ")
            if confirmation.lower() == 'o':
                replace_data()
                print("\nNouvelle analyse après remplacement:")
                analyze_data()
        elif choice == '2':
            print("\nAffichage de l'analyse actuelle:")
            analyze_data()
        elif choice == '3':
            print("\nFin du programme")
            break
        else:
            print("\nChoix non valide. Veuillez réessayer.")

if __name__ == "__main__":
    main()