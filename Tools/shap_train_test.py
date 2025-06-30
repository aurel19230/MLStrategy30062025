import pandas as pd
import glob
import os
import numpy as np
from scipy.stats import pearsonr
from colorama import init, Fore, Style, Back

init(autoreset=True)  # Initialiser colorama avec réinitialisation automatique


def analyze_files(folder_path, path_featured, file_path, importance_threshold=80):
    """
    Analyse complète des features avec SHAP values, NaN et Zeros.

    Args:
        folder_path: Chemin vers les fichiers SHAP
        path_featured: Chemin de sortie
        file_path: Chemin du fichier principal
        importance_threshold: Seuil d'importance cumulative (default: 80%)
    """
    # Vérifier l'existence du fichier principal
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier principal n'existe pas: {file_path}")

    # Charger le fichier CSV principal
    print(f"{Fore.CYAN}Chargement du fichier principal: {file_path}")
    df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    print(f"Nombre de lignes dans le fichier principal: {len(df):,}")
    print(f"Nombre de colonnes dans le fichier principal: {len(df.columns):,}")

    # Récupérer les chemins des fichiers CSV
    train_files = glob.glob(os.path.join(folder_path, '*Training_Set.csv'))
    test_files = glob.glob(os.path.join(folder_path, '*Test_Set.csv'))

    # Vérifier si des fichiers ont été trouvés
    print(f"\n{Fore.CYAN}Fichiers trouvés:")
    print(f"Training files ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
    print(f"Test files ({len(test_files)}): {[os.path.basename(f) for f in test_files]}")

    if len(train_files) == 0:
        raise ValueError(f"Aucun fichier d'entraînement trouvé dans: {folder_path}")
    if len(test_files) == 0:
        raise ValueError(f"Aucun fichier de test trouvé dans: {folder_path}")

    # Charger les fichiers dans des listes de DataFrames
    print(f"\n{Fore.CYAN}Chargement des fichiers...")
    train_dfs = []
    test_dfs = []

    for f in train_files:
        try:
            df_train = pd.read_csv(f, sep=';', encoding='latin-1')
            print(f"Chargé {os.path.basename(f)}: {len(df_train):,} lignes")
            train_dfs.append(df_train)
        except Exception as e:
            print(f"{Fore.RED}Erreur lors du chargement de {f}: {str(e)}")

    for f in test_files:
        try:
            df_test = pd.read_csv(f, sep=';', encoding='latin-1')
            print(f"Chargé {os.path.basename(f)}: {len(df_test):,} lignes")
            test_dfs.append(df_test)
        except Exception as e:
            print(f"{Fore.RED}Erreur lors du chargement de {f}: {str(e)}")

    if len(train_dfs) == 0:
        raise ValueError("Aucun DataFrame d'entraînement n'a pu être chargé")
    if len(test_dfs) == 0:
        raise ValueError("Aucun DataFrame de test n'a pu être chargé")

    # Combiner les DataFrames
    for i, df_train in enumerate(train_dfs):
        df_train['optimization'] = f'opt_{i + 1}'
    for i, df_test in enumerate(test_dfs):
        df_test['optimization'] = f'opt_{i + 1}'

    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    combined_test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"\n{Fore.CYAN}Données combinées:")
    print(f"Training: {len(combined_train_df):,} lignes")
    print(f"Test: {len(combined_test_df):,} lignes")

    # Calculer les statistiques d'importance
    mean_importance_train = combined_train_df.groupby('feature')['importance_percentage'].mean()
    mean_importance_test = combined_test_df.groupby('feature')['importance_percentage'].mean()

    # Combiner les moyennes d'importance
    combined_importance = pd.DataFrame({
        'importance_train': mean_importance_train,
        'importance_test': mean_importance_test
    })
    combined_importance['importance_mean'] = (combined_importance['importance_train'] + combined_importance[
        'importance_test']) / 2
    combined_importance = combined_importance.sort_values('importance_mean', ascending=False)

    # Calculer l'importance cumulative
    combined_importance['cumulative_importance'] = combined_importance['importance_mean'].cumsum()
    combined_importance['cumulative_importance_normalized'] = (combined_importance['cumulative_importance'] /
                                                            combined_importance['importance_mean'].sum() * 100)

    # Sélectionner les features importantes
    top_features = combined_importance[
        combined_importance['cumulative_importance_normalized'] <= importance_threshold].index.tolist()
    features_above_threshold = combined_importance[
        combined_importance['cumulative_importance_normalized'] > importance_threshold].index.tolist()

# Analyse des NaN et des zéros pour toutes les features
    all_features_stats = pd.DataFrame(index=df.columns)

    # Pour chaque colonne, calculer la répartition exacte
    for col in df.columns:
        total_rows = len(df)
        nan_count = df[col].isna().sum()

        try:
            # Convertir en numérique
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            non_nan_values = numeric_col.dropna()
            zeros_count = (non_nan_values == 0).sum()
            other_values_count = len(non_nan_values) - zeros_count

            # Calculer les pourcentages
            nan_pct = (nan_count / total_rows * 100).round(2)
            zeros_pct = (zeros_count / total_rows * 100).round(2)
            other_pct = (other_values_count / total_rows * 100).round(2)

            all_features_stats.loc[col, 'NaN (%)'] = nan_pct
            all_features_stats.loc[col, 'Zeros (%)'] = zeros_pct
            all_features_stats.loc[col, 'Other (%)'] = other_pct

        except:
            # Pour les colonnes non numériques
            all_features_stats.loc[col, 'NaN (%)'] = (nan_count / total_rows * 100).round(2)
            all_features_stats.loc[col, 'Zeros (%)'] = 0
            all_features_stats.loc[col, 'Other (%)'] = ((total_rows - nan_count) / total_rows * 100).round(2)

    # Reset index pour avoir la colonne Feature et ajouter NaN+Zeros
    all_features_stats = all_features_stats.reset_index().rename(columns={'index': 'Feature'})
    all_features_stats['NaN+Zeros (%)'] = (all_features_stats['NaN (%)'] + all_features_stats['Zeros (%)']).round(2)

    # Tri par NaN décroissant puis par Zeros décroissant
    all_features_stats = all_features_stats.sort_values(['NaN (%)', 'Zeros (%)'], ascending=[False, False])

    all_features_count = len(df.columns)

    # Statistiques sur les NaN et Zeros
    features_with_nan = (all_features_stats['NaN (%)'] > 0).sum()
    features_with_zeros = (all_features_stats['Zeros (%)'] > 0).sum()

    print(f"\n{Fore.CYAN}Statistiques NaN et Zeros:")
    print(f"Features avec NaN: {features_with_nan:,} ({features_with_nan/all_features_count*100:.2f}%)")
    print(f"Features sans NaN: {all_features_count - features_with_nan:,}")
    print(f"Features avec Zeros: {features_with_zeros:,} ({features_with_zeros/all_features_count*100:.2f}%)")

    print(f"\n{Fore.CYAN}Analyse détaillée des features")
    print("=" * 160)
    print(f"{'Feature':<50} {'NaN(%)':<10} {'Zeros(%)':<10} {'NaN+Zeros(%)':<12} {'Other(%)':<10} {'Total(%)'}")
    print("-" * 160)
    for _, row in all_features_stats.iterrows():
        if row['NaN (%)'] > 0 or row['Zeros (%)'] > 0:  # Afficher les features avec NaN ou Zeros
            try:
                total = row['NaN (%)'] + row['Zeros (%)'] + row['Other (%)']
                print(f"{str(row['Feature']):<50} "
                      f"{float(row['NaN (%)']):>8.2f}    "
                      f"{float(row['Zeros (%)']):>8.2f}    "
                      f"{float(row['NaN+Zeros (%)']):>10.2f}    "
                      f"{float(row['Other (%)']):>8.2f}    "
                      f"{total:>8.2f}")
            except Exception as e:
                print(f"Erreur de formatage pour {row['Feature']}: {str(e)}")

    # Créer DataFrame des résultats
    results = pd.DataFrame({
        'Rang': range(1, len(top_features) + 1),
        'Feature': top_features,
        'Importance (%)': combined_importance.loc[top_features, 'importance_mean'].values,
        'Importance cumulée (%)': combined_importance.loc[top_features, 'cumulative_importance_normalized'].values,
        'NaN (%)': [all_features_stats.loc[all_features_stats['Feature'] == feature, 'NaN (%)'].values[0] for feature in top_features],
        'Zeros (%)': [all_features_stats.loc[all_features_stats['Feature'] == feature, 'Zeros (%)'].values[0] for feature in top_features],
        'NaN+Zeros (%)': [all_features_stats.loc[all_features_stats['Feature'] == feature, 'NaN+Zeros (%)'].values[0] for feature in top_features],
        'Other (%)': [all_features_stats.loc[all_features_stats['Feature'] == feature, 'Other (%)'].values[0] for feature in top_features]
    })

    # Afficher les résultats
    print(f"\n{Fore.CYAN}Analyse des features importantes (seuil d'importance cumulée: {importance_threshold}%)")
    print("=" * 180)
    print(f"{'Rang':<6} {'Feature':<50} {'Importance(%)':<15} {'Cum.Imp(%)':<15} "
          f"{'NaN(%)':<10} {'Zeros(%)':<10} {'NaN+Zeros(%)':<12} {'Other(%)'}")
    print("-" * 180)

    for _, row in results.iterrows():
        try:
            print(f"{row['Rang']:<6d} {str(row['Feature']):<50} {float(row['Importance (%)']):>8.2f}       "
                  f"{float(row['Importance cumulée (%)']):>8.2f}       "
                  f"{float(row['NaN (%)']):>6.2f}    {float(row['Zeros (%)']):>6.2f}    "
                  f"{float(row['NaN+Zeros (%)']):>10.2f}    {float(row['Other (%)']):>6.2f}")
        except Exception as e:
            print(f"Erreur de formatage pour {row['Feature']}: {str(e)}")

    # Afficher features au-delà du seuil
    if features_above_threshold:
        print(f"\n{Fore.YELLOW}Features au-delà du seuil de {importance_threshold}% d'importance cumulée:")
        print("-" * 180)
        for feature in features_above_threshold:
            try:
                importance = float(combined_importance.loc[feature, 'importance_mean'])
                cumulative = float(combined_importance.loc[feature, 'cumulative_importance_normalized'])
                stats = all_features_stats.loc[all_features_stats['Feature'] == feature].iloc[0]
                print(f"{'--':<6} {feature:<50} {importance:>8.2f}       {cumulative:>8.2f}       "
                      f"{float(stats['NaN (%)']):>6.2f}    {float(stats['Zeros (%)']):>6.2f}    "
                      f"{float(stats['NaN+Zeros (%)']):>10.2f}    {float(stats['Other (%)']):>6.2f}")
            except Exception as e:
                print(f"Erreur de formatage pour {feature}: {str(e)}")

    print(f"\n{Fore.CYAN}Statistiques supplémentaires:")
    print(f"Seuil d'importance cumulée: {importance_threshold}%")
    print(f"Nombre total de features sélectionnées: {len(top_features):,}")
    print(f"Nombre de features au-delà du seuil: {len(features_above_threshold):,}")
    print(f"Importance totale cumulée des features sélectionnées: {combined_importance.loc[top_features, 'importance_mean'].sum():.2f}%")
    print(f"Pourcentage de features sélectionnées: {(len(top_features)/all_features_count)*100:.2f}%")

    # Sauvegarder les features sélectionnées
    output_file = os.path.join(path_featured, 'selected_features.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for feature in top_features:
            f.write(f"{feature}\n")

    print(f"\nLes features sélectionnées ont été sauvegardées dans {output_file}")

    return results, combined_importance, all_features_stats


if __name__ == "__main__":
    # Chemins des fichiers
    folder_path = r'C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\results_optim\4_0_4TP_1SL_10_30_291024 sans big et les nb nan ou 0\shap_dependencies_results'
    path_featured = r'C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge'
    file_path = os.path.join(path_featured,
                            'Step5_4_0_4TP_1SL_080919_091024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv')

    # Définir le seuil d'importance
    IMPORTANCE_THRESHOLD = 80

    try:
        results, combined_importance, all_features_stats = analyze_files(
            folder_path,
            path_featured,
            file_path,
            importance_threshold=IMPORTANCE_THRESHOLD
        )
    except Exception as e:
        print(f"{Fore.RED}Une erreur s'est produite: {str(e)}")
